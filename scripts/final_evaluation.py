import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.evaluation.metrics_reporter import report_all_metrics
from src.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR, PROJECT_ROOT
from src.utils.bio_encoder import BioFeatureEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Final Evaluation on {device}...")

    # 1. Load Data & Supports
    val_path = PROCESSED_DATA_DIR / "val.csv"
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    if not all(p.exists() for p in [val_path, emb_path, map_path, graph_path]):
        print("Missing required data files (val.csv, embeddings.pt, mapping, or graph).")
        return

    val_df = pd.read_csv(val_path)
    embeddings = torch.load(emb_path, weights_only=False)
    # Convert half to float
    embeddings = {k: v.float().cpu() if v.dtype == torch.float16 else v.cpu() for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)
    graph_data = torch.load(graph_path, weights_only=False).to(device)

    # Filter val_df
    filtered_df = val_df[
        val_df["protein1"].isin(embeddings) & 
        val_df["protein2"].isin(embeddings) &
        val_df["protein1"].isin(node_mapping) &
        val_df["protein2"].isin(node_mapping)
    ].copy()
    
    y_true = filtered_df["label"].values
    
    # 2. Load Models
    print("Loading Models...")
    
    # Dynamically detect input_dim
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1]
    
    # Load bio-features for dimension detection
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0
    print(f"Detected Dimensions: Sequence={input_dim}, Biology={bio_dim}")

    seq_model = SequencePPIModel(input_dim=input_dim, bio_dim=bio_dim).to(device)
    seq_model.load_state_dict(torch.load(MODELS_DIR / "sequence_model_best.pth", map_location=device))
    seq_model.eval()

    # Graph features are already injected properly by extract_topo in run_pipeline.py

    from src.models.graph_model import GINLinkPredictor
    state_dict = torch.load(MODELS_DIR / "graph_model_best.pth", map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    
    if is_gin:
        graph_model = GINLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128).to(device)
    else:
        graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128, heads=4).to(device)
        
    graph_model.load_state_dict(state_dict)
    graph_model.eval()

    ensemble = PPIEnsemble(str(MODELS_DIR / "ensemble_model.pkl"))

    # 3. Generate Predictions
    print("Generating predictions...")
    
    # --- BIOLOGICAL INJECTION (Localization Match) ---
    bio_path = PROCESSED_DATA_DIR / "bio_metadata_cache.csv"
    bio_df = pd.read_csv(bio_path).set_index("protein_id")["localization"].to_dict() if bio_path.exists() else {}

    batch_emb1 = []
    batch_emb2 = []
    g_src = []
    g_dst = []
    bio_features = []
    
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Preparing Data"):
        p1, p2 = row["protein1"], row["protein2"]
        e1, e2 = embeddings[p1], embeddings[p2]
        e1_mean = e1.mean(dim=0) if e1.dim() > 1 else e1
        e2_mean = e2.mean(dim=0) if e2.dim() > 1 else e2
        
        if bio_mapping:
            b1 = bio_mapping.get(p1, torch.zeros(bio_dim))
            b2 = bio_mapping.get(p2, torch.zeros(bio_dim))
            e1_mean = torch.cat([e1_mean, b1])
            e2_mean = torch.cat([e2_mean, b2])
            
        batch_emb1.append(e1_mean)
        batch_emb2.append(e2_mean)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])

        # Localization match feature
        loc1, loc2 = bio_df.get(p1, "unk1"), bio_df.get(p2, "unk2")
        bio_features.append([1.0 if loc1 == loc2 and loc1 != "unk1" else 0.0])

    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)
    bio_features = np.array(bio_features)
    
    seq_probs = []
    with torch.no_grad():
        for i in range(0, len(batch_emb1), 64):
            e1 = batch_emb1[i:i+64].to(device)
            e2 = batch_emb2[i:i+64].to(device)
            out = seq_model(e1, e2)
            seq_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    seq_probs = np.array(seq_probs)

    # Graph Model
    g_edge_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    graph_probs = []
    with torch.no_grad():
        for i in range(0, g_edge_index.size(1), 10000):
            chunk = g_edge_index[:, i:i+10000]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            graph_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    graph_probs = np.array(graph_probs)

    # Ensemble
    ensemble_probs = ensemble.predict(seq_probs, graph_probs, bio_features=bio_features, method="stacking")

    # 4. Report
    report_all_metrics(
        ["ESM-MLP", "GAT", "Ensemble"],
        [y_true, y_true, y_true],
        [seq_probs, graph_probs, ensemble_probs]
    )

if __name__ == "__main__":
    main()
