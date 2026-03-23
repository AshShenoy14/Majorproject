"""Diagnostic script to confirm embedding mismatch between training and inference."""
import torch, sys, os, pandas as pd
sys.path.append('.')
from src.models.sequence_model import SequencePPIModel
from src.data.feature_extraction import ESMFeatureExtractor
from src.utils.paths import PROJECT_ROOT, PROCESSED_DATA_DIR

device = "cpu"

# Auto-detect embedding dimension from embeddings
embs_peek = torch.load(PROCESSED_DATA_DIR / "embeddings.pt", weights_only=False)
sample_emb = next(iter(embs_peek.values()))
input_dim = sample_emb.shape[-1] if sample_emb.dim() > 1 else sample_emb.shape[0]
del embs_peek
print(f"Detected embedding dimension: {input_dim}")

model = SequencePPIModel(input_dim=input_dim).to(device)
seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
model.load_state_dict(torch.load(seq_path, map_location=device, weights_only=True))
model.eval()

# 2. Get a protein pair from train.csv
df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
pos_row = df[df["label"]==1].iloc[0]
neg_row = df[df["label"]==0].iloc[0]
p1, p2 = pos_row["protein1"], pos_row["protein2"]
n1, n2 = neg_row["protein1"], neg_row["protein2"]
print(f"Positive pair: {p1} - {p2}")
print(f"Negative pair: {n1} - {n2}")

# 3. Load pre-computed embeddings and do simple mean (like training)
print("\nLoading embeddings.pt ...")
embs = torch.load(PROCESSED_DATA_DIR / "embeddings.pt", weights_only=False)

# Check shape of first embedding
first_key = list(embs.keys())[0]
print(f"Sample embedding shape: {embs[first_key].shape} (dtype: {embs[first_key].dtype})")

def pool_like_training(emb):
    """Same pooling as PPIDataset.__getitem__"""
    e = emb.float()
    if e.dim() > 1:
        e = e.mean(dim=0)
    return e

e1_train = pool_like_training(embs[p1])
e2_train = pool_like_training(embs[p2])
n1_train = pool_like_training(embs[n1])
n2_train = pool_like_training(embs[n2])
print(f"After training-style pooling shape: {e1_train.shape}")
print(f"  Pos e1 stats: mean={e1_train.mean():.4f}, std={e1_train.std():.4f}")

# 4. Get inference-time embeddings (current code)
from src.data.sequence_manager import SequenceManager
mgr = SequenceManager()
all_proteins = [p1, p2, n1, n2]
seqs = mgr.get_sequences(all_proteins)
print(f"\nGot sequences for: {list(seqs.keys())}")

extractor = ESMFeatureExtractor(device=device)
inf_embs = extractor.get_embeddings(seqs, batch_size=4)
e1_inf = inf_embs[p1].float()
e2_inf = inf_embs[p2].float()
n1_inf = inf_embs[n1].float()
n2_inf = inf_embs[n2].float()
print(f"Inference emb shape: {e1_inf.shape}")
print(f"  Pos e1 stats: mean={e1_inf.mean():.4f}, std={e1_inf.std():.4f}")

# 5. Compare embeddings
diff = (e1_train - e1_inf).abs()
print(f"\nEmbedding difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

# 6. Run model with BOTH kinds of embeddings
with torch.no_grad():
    # Positive pair
    logit_train_pos = model(e1_train.unsqueeze(0), e2_train.unsqueeze(0)).item()
    logit_inf_pos = model(e1_inf.unsqueeze(0), e2_inf.unsqueeze(0)).item()
    prob_train_pos = torch.sigmoid(torch.tensor(logit_train_pos)).item()
    prob_inf_pos = torch.sigmoid(torch.tensor(logit_inf_pos)).item()
    
    # Negative pair
    logit_train_neg = model(n1_train.unsqueeze(0), n2_train.unsqueeze(0)).item()
    logit_inf_neg = model(n1_inf.unsqueeze(0), n2_inf.unsqueeze(0)).item()
    prob_train_neg = torch.sigmoid(torch.tensor(logit_train_neg)).item()
    prob_inf_neg = torch.sigmoid(torch.tensor(logit_inf_neg)).item()

print("\n" + "="*60)
print("POSITIVE PAIR (should predict ~1.0):")
print(f"  Training-style emb: logit={logit_train_pos:.4f}, prob={prob_train_pos:.4f}")
print(f"  Inference-style emb: logit={logit_inf_pos:.4f}, prob={prob_inf_pos:.4f}")
print()
print("NEGATIVE PAIR (should predict ~0.0):")
print(f"  Training-style emb: logit={logit_train_neg:.4f}, prob={prob_train_neg:.4f}")
print(f"  Inference-style emb: logit={logit_inf_neg:.4f}, prob={prob_inf_neg:.4f}")
print("="*60)

# 7. Now test with SIMPLE MEAN on raw ESM output (matching training)
print("\n--- Testing simple mean pooling on raw ESM output ---")
inputs = extractor.tokenizer(
    [seqs[p1], seqs[p2], seqs[n1], seqs[n2]],
    return_tensors="pt", padding=True, truncation=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = extractor.model(**inputs)
    # Simple mean (matching training)
    simple_mean_embs = outputs.last_hidden_state.mean(dim=1)

e1_simple = simple_mean_embs[0]
e2_simple = simple_mean_embs[1]
n1_simple = simple_mean_embs[2]
n2_simple = simple_mean_embs[3]

with torch.no_grad():
    logit_simple_pos = model(e1_simple.unsqueeze(0), e2_simple.unsqueeze(0)).item()
    logit_simple_neg = model(n1_simple.unsqueeze(0), n2_simple.unsqueeze(0)).item()
    prob_simple_pos = torch.sigmoid(torch.tensor(logit_simple_pos)).item()
    prob_simple_neg = torch.sigmoid(torch.tensor(logit_simple_neg)).item()

print(f"POSITIVE with simple mean: logit={logit_simple_pos:.4f}, prob={prob_simple_pos:.4f}")
print(f"NEGATIVE with simple mean: logit={logit_simple_neg:.4f}, prob={prob_simple_neg:.4f}")

# Also compare simple mean vs stored
diff_simple = (e1_train - e1_simple).abs()
print(f"\nDiff (stored vs simple_mean): mean={diff_simple.mean():.6f}, max={diff_simple.max():.6f}")
diff_attn = (e1_train - e1_inf).abs()
print(f"Diff (stored vs attn_mean):   mean={diff_attn.mean():.6f}, max={diff_attn.max():.6f}")
