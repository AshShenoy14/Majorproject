import numpy as np
import torch
import pandas as pd
import argparse
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def train_ensemble(seq_model_path, graph_model_path, graph_data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}...")
    
    # 1. Load Base Models
    print("Loading base models...")
    
    # Sequence Model
    input_dim = 320 
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    try:
        if os.path.exists(seq_model_path):
            seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
            print(f"Loaded Sequence Model from {seq_model_path}")
        else:
            print(f"Sequence model not found at {seq_model_path}")
            return
    except Exception as e:
        print(f"Failed to load sequence model state dict: {e}")
        return
    seq_model.eval()

    # Graph Model — updated to match new architecture (hidden=128)
    if not os.path.exists(graph_data_path):
        print(f"Graph data not found at {graph_data_path}")
        return
    
    graph_data = torch.load(graph_data_path, weights_only=False).to(device)
    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=64).to(device)
    try:
        if os.path.exists(graph_model_path):
            graph_model.load_state_dict(torch.load(graph_model_path, map_location=device))
            print(f"Loaded Graph Model from {graph_model_path}")
        else:
            print(f"Graph model not found at {graph_model_path}")
            return
    except Exception as e:
        print(f"Failed to load graph model state dict: {e}")
        return
    graph_model.eval()

    # 2. Generate Predictions on Validation Set
    print("Generating predictions on Validation Set...")
    val_path = PROCESSED_DATA_DIR / "val.csv"
    if not val_path.exists():
        print(f"Validation data not found at {val_path}")
        return

    val_df = pd.read_csv(val_path)
    
    # Load support files
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    
    if not emb_path.exists() or not map_path.exists():
        print("Embeddings or Mapping not found. Cannot run inference.")
        return
        
    embeddings = torch.load(emb_path, weights_only=False)
    # Convert float16 embeddings to float32 for model compatibility
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)
    
    # Filter val_df to valid entries
    filtered_df = val_df[
        val_df["protein1"].isin(embeddings) & 
        val_df["protein2"].isin(embeddings) &
        val_df["protein1"].isin(node_mapping) &
        val_df["protein2"].isin(node_mapping)
    ].copy()
    
    final_labels = []
    
    # Prepare batch data for Sequence Model
    batch_emb1 = []
    batch_emb2 = []
    
    # Prepare indices for Graph Model
    g_src = []
    g_dst = []
    
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Aligning Data"):
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        
        # Seq components — mean-pool per-residue embeddings to fixed-size vectors
        e1 = embeddings[p1]
        e2 = embeddings[p2]
        batch_emb1.append(e1.mean(dim=0) if e1.dim() > 1 else e1)
        batch_emb2.append(e2.mean(dim=0) if e2.dim() > 1 else e2)
        
        # Graph components
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        
        final_labels.append(label)

    # Run Sequence Model
    print("Predicting with Sequence Model...")
    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)
    batch_size = 32
    
    final_seq_preds = []
    with torch.no_grad():
        for i in range(0, len(batch_emb1), batch_size):
            e1 = batch_emb1[i:i+batch_size].to(device)
            e2 = batch_emb2[i:i+batch_size].to(device)
            out = seq_model(e1, e2)
            # Apply sigmoid to raw logits (model no longer has built-in Sigmoid)
            probs = torch.sigmoid(out)
            final_seq_preds.extend(probs.cpu().numpy().flatten())
            
    # Run Graph Model — apply sigmoid to raw logits
    print("Predicting with Graph Model...")
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    
    final_graph_preds = []
    with torch.no_grad():
        chunk_size = 10000
        for i in range(0, g_edge_label_index.size(1), chunk_size):
            chunk = g_edge_label_index[:, i:i+chunk_size]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            # Apply sigmoid to raw logits
            probs = torch.sigmoid(out)
            final_graph_preds.extend(probs.cpu().numpy().flatten())

    val_labels_np = np.array(final_labels)
    seq_preds_np = np.array(final_seq_preds)
    graph_preds_np = np.array(final_graph_preds)
    
    if len(val_labels_np) == 0:
        print("No valid validation samples found.")
        return

    # 3. Train Ensemble with enhanced features
    ensemble = PPIEnsemble()
    ensemble.train_stacking(seq_preds_np, graph_preds_np, val_labels_np)
    
    # Generate Ensemble Predictions
    ensemble_preds = ensemble.predict(seq_preds_np, graph_preds_np, method="stacking")
    
    # Save
    out_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    ensemble.save(out_path)
    print("Ensemble training complete.")
    
    # 4. Generate ROC-AUC and Precision-Recall Curves
    print("Generating ROC-AUC and Precision-Recall Curves...")
    assets_dir = PROJECT_ROOT / "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # --- ROC Curve ---
    plt.figure(figsize=(8, 6))
    
    # Sequence Model ROC
    fpr_seq, tpr_seq, _ = roc_curve(val_labels_np, seq_preds_np)
    roc_auc_seq = auc(fpr_seq, tpr_seq)
    plt.plot(fpr_seq, tpr_seq, color='blue', lw=2, label=f'Sequence Model (AUC = {roc_auc_seq:.3f})')
    
    # Graph Model ROC
    fpr_graph, tpr_graph, _ = roc_curve(val_labels_np, graph_preds_np)
    roc_auc_graph = auc(fpr_graph, tpr_graph)
    plt.plot(fpr_graph, tpr_graph, color='green', lw=2, label=f'Graph Model (AUC = {roc_auc_graph:.3f})')
    
    # Ensemble Model ROC
    fpr_ens, tpr_ens, _ = roc_curve(val_labels_np, ensemble_preds)
    roc_auc_ens = auc(fpr_ens, tpr_ens)
    plt.plot(fpr_ens, tpr_ens, color='red', lw=2, label=f'Ensemble Model (AUC = {roc_auc_ens:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_path = assets_dir / 'roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC Curve saved to {roc_path}")
    
    # --- Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    
    # Sequence Model PR
    precision_seq, recall_seq, _ = precision_recall_curve(val_labels_np, seq_preds_np)
    ap_seq = average_precision_score(val_labels_np, seq_preds_np)
    plt.plot(recall_seq, precision_seq, color='blue', lw=2, label=f'Sequence Model (AP = {ap_seq:.3f})')
    
    # Graph Model PR
    precision_graph, recall_graph, _ = precision_recall_curve(val_labels_np, graph_preds_np)
    ap_graph = average_precision_score(val_labels_np, graph_preds_np)
    plt.plot(recall_graph, precision_graph, color='green', lw=2, label=f'Graph Model (AP = {ap_graph:.3f})')
    
    # Ensemble Model PR
    precision_ens, recall_ens, _ = precision_recall_curve(val_labels_np, ensemble_preds)
    ap_ens = average_precision_score(val_labels_np, ensemble_preds)
    plt.plot(recall_ens, precision_ens, color='red', lw=2, label=f'Ensemble Model (AP = {ap_ens:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    pr_path = assets_dir / 'pr_curve.png'
    plt.savefig(pr_path)
    plt.close()
    print(f"Precision-Recall Curve saved to {pr_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Define default paths
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    graph_data = PROCESSED_DATA_DIR / "ppi_graph.pt"
    
    train_ensemble(seq_path, graph_path, graph_data)
