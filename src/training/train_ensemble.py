import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor, GINLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR
from src.utils.bio_encoder import BioFeatureEncoder
from src.analysis.biological_managers import BiologicalManager


def load_base_models(seq_model_path, graph_model_path, graph_data, input_dim, in_channels, device):
    """Loads full-dataset base Sequence and Graph models from checkpoints (used ONLY for final validation evaluation)."""
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    if os.path.exists(seq_model_path):
        seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
        print(f"Loaded Final Sequence Model from {seq_model_path}")
    else:
        raise FileNotFoundError(f"Sequence model checkpoint not found at {seq_model_path}")
    seq_model.eval()

    state_dict = torch.load(graph_model_path, map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    if is_gin:
        print("Detected GIN architecture for Graph Model.")
        graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
    else:
        print("Detected GAT architecture for Graph Model.")
        graph_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
    
    graph_model.load_state_dict(state_dict)
    print(f"Loaded Final Graph Model from {graph_model_path}")
    graph_model.eval()

    return seq_model, graph_model


def predict_sequence_model(model, embeddings, bio_mapping, p1_list, p2_list, device, bio_dim=0, batch_size=64):
    """Runs batch inference for Sequence Model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(p1_list), batch_size):
            p1_batch = p1_list[i:i+batch_size]
            p2_batch = p2_list[i:i+batch_size]
            
            b_emb1, b_emb2 = [], []
            for p1, p2 in zip(p1_batch, p2_batch):
                e1 = embeddings[p1].float()
                e2 = embeddings[p2].float()
                e1_v = e1.mean(dim=0) if e1.dim() > 1 else e1
                e2_v = e2.mean(dim=0) if e2.dim() > 1 else e2
                if bio_mapping:
                    b1 = bio_mapping.get(p1, torch.zeros(bio_dim))
                    b2 = bio_mapping.get(p2, torch.zeros(bio_dim))
                    e1_v = torch.cat([e1_v, b1])
                    e2_v = torch.cat([e2_v, b2])
                b_emb1.append(e1_v)
                b_emb2.append(e2_v)
            
            e1_t = torch.stack(b_emb1).to(device)
            e2_t = torch.stack(b_emb2).to(device)
            out = model(e1_t, e2_t)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds.extend(probs)
    return np.array(preds)


def predict_graph_model(graph_model, fold_graph_data, node_mapping, p1_list, p2_list, device, chunk_size=1000):
    """Runs batch inference for Graph Model."""
    graph_model.eval()
    src_indices = [node_mapping[p1] for p1 in p1_list]
    dst_indices = [node_mapping[p2] for p2 in p2_list]
    edge_label_index = torch.tensor([src_indices, dst_indices], dtype=torch.long).to(device)
    
    preds = []
    with torch.no_grad():
        z = graph_model.encode(fold_graph_data.x, fold_graph_data.edge_index)
        num_edges = edge_label_index.size(1)
        for i in range(0, num_edges, chunk_size):
            chunk = edge_label_index[:, i:i+chunk_size]
            out = graph_model.decode(z, chunk[0], chunk[1])
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds.extend(probs)
    return np.array(preds)


def train_fold_sequence(model, embeddings, bio_mapping, train_p1, train_p2, train_labels, device, bio_dim=0, epochs=5, batch_size=64):
    """Trains a Sequence PPI Model from scratch on a fold's training split."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    num_samples = len(train_p1)

    for ep in range(epochs):
        perm = np.random.permutation(num_samples)
        for b in range(0, num_samples, batch_size):
            batch_ids = perm[b:b+batch_size]
            b_p1, b_p2, b_lbl = train_p1[batch_ids], train_p2[batch_ids], train_labels[batch_ids]
            
            b_e1, b_e2 = [], []
            for p1, p2 in zip(b_p1, b_p2):
                e1 = embeddings[p1].float()
                e2 = embeddings[p2].float()
                e1_v = e1.mean(dim=0) if e1.dim() > 1 else e1
                e2_v = e2.mean(dim=0) if e2.dim() > 1 else e2
                if bio_mapping:
                    b1 = bio_mapping.get(p1, torch.zeros(bio_dim))
                    b2 = bio_mapping.get(p2, torch.zeros(bio_dim))
                    e1_v = torch.cat([e1_v, b1])
                    e2_v = torch.cat([e2_v, b2])
                b_e1.append(e1_v)
                b_e2.append(e2_v)
            
            e1_t = torch.stack(b_e1).to(device)
            e2_t = torch.stack(b_e2).to(device)
            lbl_t = torch.tensor(b_lbl, dtype=torch.float32).to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(e1_t, e2_t)
            loss = criterion(out, lbl_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def train_fold_gat(model, fold_graph_data, train_p1, train_p2, train_labels, node_mapping, device, epochs=5, chunk_size=1000):
    """Trains a GAT Link Predictor from scratch on a fold's training graph and pairs."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    src = [node_mapping[p1] for p1 in train_p1]
    dst = [node_mapping[p2] for p2 in train_p2]
    edge_label_index = torch.tensor([src, dst], dtype=torch.long).to(device)
    labels_t = torch.tensor(train_labels, dtype=torch.float32).to(device)
    num_edges = edge_label_index.size(1)

    for ep in range(epochs):
        optimizer.zero_grad()
        z = model.encode(fold_graph_data.x, fold_graph_data.edge_index)
        z_detached = z.detach().requires_grad_(True)
        
        for i in range(0, num_edges, chunk_size):
            chunk = edge_label_index[:, i:i+chunk_size]
            lbl_c = labels_t[i:i+chunk_size]
            out_c = model.decode(z_detached, chunk[0], chunk[1])
            loss_c = criterion(out_c.squeeze(), lbl_c)
            (loss_c * 10.0).backward()
            
        z.backward(z_detached.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def generate_oof_predictions(
    train_df, 
    embeddings, 
    bio_mapping, 
    node_mapping, 
    full_graph_data, 
    device, 
    k_folds=5, 
    oof_epochs=5,
    dry_run=False
):
    """
    Generates STRICT, leakage-free out-of-fold (OOF) predictions on train.csv.
    Each fold initializes base models from scratch and trains strictly on the in-fold training portion.
    Full-data checkpoints (sequence_model_best.pth / graph_model_best.pth) are NEVER loaded here.
    """
    print(f"\n--- Generating Strict Leakage-Free OOF Predictions ({k_folds} folds, {oof_epochs} epochs/fold) ---")
    
    n_samples = len(train_df)
    oof_seq_preds = np.zeros(n_samples, dtype=np.float32)
    oof_graph_preds = np.zeros(n_samples, dtype=np.float32)
    visited_indices = np.zeros(n_samples, dtype=bool)
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    p1_all = train_df["protein1"].values
    p2_all = train_df["protein2"].values
    labels_all = train_df["label"].values

    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1]
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0
    in_channels = full_graph_data.x.shape[1]

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, labels_all)):
        print(f"\nProcessing Fold {fold+1}/{k_folds} (In-Fold Train: {len(train_idx)}, Held-Out OOF: {len(val_idx)})...")
        
        # 1. Assert disjoint splits
        assert set(train_idx).isdisjoint(set(val_idx)), f"Fold {fold+1}: Train and validation indices overlap!"
        
        # 2. Construct Fold-Specific GAT Graph using ONLY positive training edges from in-fold train split
        in_fold_train_df = train_df.iloc[train_idx]
        pos_train_df = in_fold_train_df[in_fold_train_df["label"] == 1]
        
        fold_src_nodes = [node_mapping[p] for p in pos_train_df["protein1"]]
        fold_dst_nodes = [node_mapping[p] for p in pos_train_df["protein2"]]
        
        # Undirected graph edge tensor
        all_fold_src = fold_src_nodes + fold_dst_nodes
        all_fold_dst = fold_dst_nodes + fold_src_nodes
        fold_edge_index = torch.tensor([all_fold_src, all_fold_dst], dtype=torch.long).to(device)
        
        # Explicit verification: ensure no held-out validation edge is present in fold GAT graph
        val_src = [node_mapping[p] for p in train_df.iloc[val_idx]["protein1"]]
        val_dst = [node_mapping[p] for p in train_df.iloc[val_idx]["protein2"]]
        val_pairs_set = set(zip(val_src, val_dst)).union(set(zip(val_dst, val_src)))
        fold_edge_pairs_set = set(zip(all_fold_src, all_fold_dst))
        
        leaked_edges = val_pairs_set.intersection(fold_edge_pairs_set)
        assert len(leaked_edges) == 0, f"LEAKAGE DETECTED in Fold {fold+1}: {len(leaked_edges)} held-out edges found in fold graph!"
        print(f"  [VERIFIED] Zero held-out validation edges in Fold {fold+1} GAT graph.")

        fold_graph_data = Data(x=full_graph_data.x.clone(), edge_index=fold_edge_index)

        # 3. Instantiate fresh models from scratch (Zero checkpoint fallback)
        print(f"  Initializing fresh Sequence and GAT models from scratch for Fold {fold+1}...")
        fold_seq_model = SequencePPIModel(input_dim=input_dim).to(device)
        fold_gat_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)

        # 4. Train Fold Models strictly on in-fold training data
        tr_p1, tr_p2, tr_lbl = p1_all[train_idx], p2_all[train_idx], labels_all[train_idx]
        print(f"  Training Fold {fold+1} Sequence Model ({oof_epochs} epochs)...")
        train_fold_sequence(fold_seq_model, embeddings, bio_mapping, tr_p1, tr_p2, tr_lbl, device, bio_dim=bio_dim, epochs=oof_epochs)
        
        print(f"  Training Fold {fold+1} GAT Model ({oof_epochs} epochs)...")
        train_fold_gat(fold_gat_model, fold_graph_data, tr_p1, tr_p2, tr_lbl, node_mapping, device, epochs=oof_epochs)

        # 5. Predict ONLY on held-out validation fold (val_idx)
        val_p1 = p1_all[val_idx]
        val_p2 = p2_all[val_idx]
        
        print(f"  Predicting held-out OOF samples for Fold {fold+1}...")
        fold_seq_preds = predict_sequence_model(fold_seq_model, embeddings, bio_mapping, val_p1, val_p2, device, bio_dim=bio_dim)
        fold_graph_preds = predict_graph_model(fold_gat_model, fold_graph_data, node_mapping, val_p1, val_p2, device)
        
        oof_seq_preds[val_idx] = fold_seq_preds
        oof_graph_preds[val_idx] = fold_graph_preds
        visited_indices[val_idx] = True

        if dry_run:
            print(f"[DRY-RUN] Stopping after Fold 1 verification pass.")
            break

    # 6. Post-Generation Coverage & Dimension Verification
    if not dry_run:
        assert visited_indices.all(), "ERROR: Not all train.csv samples received an OOF prediction!"
        assert oof_seq_preds.shape[0] == n_samples, f"OOF sequence prediction shape mismatch ({oof_seq_preds.shape[0]} vs {n_samples})!"
        assert oof_graph_preds.shape[0] == n_samples, f"OOF graph prediction shape mismatch ({oof_graph_preds.shape[0]} vs {n_samples})!"
        print(f"\n[VERIFIED] All {n_samples} training samples received exactly 1 leakage-free OOF prediction.")

    return oof_seq_preds, oof_graph_preds, visited_indices


def train_ensemble(seq_model_path, graph_model_path, graph_data_path, k_folds=5, oof_epochs=5, dry_run=False, max_samples=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Ensemble Training Pipeline on {device}...")

    # 1. Load Support Files
    print("Loading support files (embeddings, graph data, node mapping, bio features)...")
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()

    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    if not emb_path.exists() or not map_path.exists() or not os.path.exists(graph_data_path):
        raise FileNotFoundError("Required embeddings, node mapping, or graph data file not found.")

    embeddings = torch.load(emb_path, weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)
    full_graph_data = torch.load(graph_data_path, weights_only=False).to(device)

    # 2. Load Train Dataset for Meta-Learner Training
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    
    # Filter train_df to valid entries present in mapping & embeddings
    filtered_train_df = train_df[
        train_df["protein1"].isin(embeddings) & 
        train_df["protein2"].isin(embeddings) &
        train_df["protein1"].isin(node_mapping) &
        train_df["protein2"].isin(node_mapping)
    ].copy().reset_index(drop=True)

    if dry_run and max_samples:
        print(f"[DRY-RUN] Subsampling train data to {max_samples} samples...")
        filtered_train_df = filtered_train_df.iloc[:max_samples].copy().reset_index(drop=True)

    print(f"Meta-Learner Training Dataset: {len(filtered_train_df)} samples from train.csv.")

    # 3. Generate Leakage-Free OOF Base-Model Predictions on train.csv
    oof_seq_preds, oof_graph_preds, visited_indices = generate_oof_predictions(
        filtered_train_df,
        embeddings,
        bio_mapping,
        node_mapping,
        full_graph_data,
        device,
        k_folds=k_folds,
        oof_epochs=oof_epochs,
        dry_run=dry_run
    )

    if dry_run:
        print("\n[DRY-RUN VERIFICATION COMPLETE] OOF pipeline structure and leakage assertions passed successfully.")
        return

    # 4. Calculate Biological Compatibility Scores for Train Set
    print("Calculating biological compatibility scores for training set...")
    bio_manager = BiologicalManager()
    train_bio_scores = []
    for _, row in tqdm(filtered_train_df.iterrows(), total=len(filtered_train_df), desc="Bio Analysis (Train)"):
        p1, p2 = row["protein1"], row["protein2"]
        comp = bio_manager.check_localization_compatibility(p1, p2, fetch_missing=False)
        train_bio_scores.append(comp.get("score", 0.5))
    
    train_bio_scores_np = np.array(train_bio_scores).reshape(-1, 1)
    train_labels_np = filtered_train_df["label"].values

    # 5. Train XGBoost Meta-Learner strictly on OOF predictions and train labels
    print("\n--- Fitting XGBoost Meta-Learner ---")
    ensemble = PPIEnsemble()
    ensemble.train_stacking(oof_seq_preds, oof_graph_preds, train_labels_np, bio_features=train_bio_scores_np)

    # Save Meta-Learner Model
    out_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    os.makedirs(out_path.parent, exist_ok=True)
    ensemble.save(out_path)
    print("Ensemble meta-learner training and save complete.")

    # 6. Separate Evaluation on Held-Out Validation Set (val.csv)
    print("\n--- Evaluating Trained Ensemble on Validation Set (val.csv) ---")
    val_path = PROCESSED_DATA_DIR / "val.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        filtered_val_df = val_df[
            val_df["protein1"].isin(embeddings) & 
            val_df["protein2"].isin(embeddings) &
            val_df["protein1"].isin(node_mapping) &
            val_df["protein2"].isin(node_mapping)
        ].copy().reset_index(drop=True)

        print(f"Validation Dataset: {len(filtered_val_df)} samples from val.csv.")
        
        # Load final trained base models for validation evaluation ONLY
        sample_emb = next(iter(embeddings.values()))
        input_dim = sample_emb.shape[-1]
        in_channels = full_graph_data.x.shape[1]
        seq_model, graph_model = load_base_models(seq_model_path, graph_model_path, full_graph_data, input_dim, in_channels, device)

        val_p1 = filtered_val_df["protein1"].values
        val_p2 = filtered_val_df["protein2"].values
        val_labels_np = filtered_val_df["label"].values

        val_seq_preds = predict_sequence_model(seq_model, embeddings, bio_mapping, val_p1, val_p2, device, bio_dim=len(next(iter(bio_mapping.values()))) if bio_mapping else 0)
        val_graph_preds = predict_graph_model(graph_model, full_graph_data, node_mapping, val_p1, val_p2, device)

        val_bio_scores = []
        for _, row in filtered_val_df.iterrows():
            comp = bio_manager.check_localization_compatibility(row["protein1"], row["protein2"], fetch_missing=False)
            val_bio_scores.append(comp.get("score", 0.5))
        val_bio_scores_np = np.array(val_bio_scores).reshape(-1, 1)

        val_ensemble_preds = ensemble.predict(val_seq_preds, val_graph_preds, bio_features=val_bio_scores_np, method="stacking")

        acc_seq = accuracy_score(val_labels_np, (val_seq_preds > 0.5).astype(int))
        acc_graph = accuracy_score(val_labels_np, (val_graph_preds > 0.5).astype(int))
        acc_ens = accuracy_score(val_labels_np, (val_ensemble_preds > 0.5).astype(int))
        
        auc_seq = roc_auc_score(val_labels_np, val_seq_preds)
        auc_graph = roc_auc_score(val_labels_np, val_graph_preds)
        auc_ens = roc_auc_score(val_labels_np, val_ensemble_preds)

        print(f"\n--- Validation Set Performance (val.csv) ---")
        print(f"Sequence Model Acc: {acc_seq*100:.2f}% | ROC-AUC: {auc_seq:.4f}")
        print(f"Graph Model Acc:    {acc_graph*100:.2f}% | ROC-AUC: {auc_graph:.4f}")
        print(f"Ensemble Model Acc: {acc_ens*100:.2f}% | ROC-AUC: {auc_ens:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ensemble Stacking Meta-Learner using OOF Predictions.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for OOF generation.")
    parser.add_argument("--oof_epochs", type=int, default=5, help="Epochs to train each base model from scratch per fold.")
    parser.add_argument("--dry_run", action="store_true", help="Run fast dry-run verification (1 fold, subsampled).")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum samples for dry-run verification.")
    args = parser.parse_args()

    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    graph_data = PROCESSED_DATA_DIR / "ppi_graph.pt"

    train_ensemble(
        seq_path, 
        graph_path, 
        graph_data, 
        k_folds=args.k_folds, 
        oof_epochs=args.oof_epochs, 
        dry_run=args.dry_run, 
        max_samples=args.max_samples
    )
