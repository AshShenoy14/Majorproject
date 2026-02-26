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

def train_ensemble(seq_model_path, graph_model_path, graph_data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}...")
    
    # 1. Load Base Models
    print("Loading base models...")
    
    # Sequence Model
    # We need to know input_dim. Let's assume 320 for ESM-2 t6_8M
    # Ideally this should be saved with the model or config
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

    # Graph Model
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
    
    seq_preds = []
    labels = []
    
    # Sequence Model Inference Loop
    print("Running Sequence Model Inference...")
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(val_df), batch_size)):
            batch = val_df.iloc[i:i+batch_size]
            b_emb1 = []
            b_emb2 = []
            b_labels = []
            valid_indices = []

            for idx, row in batch.iterrows():
                p1, p2, label = row["protein1"], row["protein2"], row["label"]
                if p1 in embeddings and p2 in embeddings:
                    b_emb1.append(embeddings[p1])
                    b_emb2.append(embeddings[p2])
                    b_labels.append(label)
                    valid_indices.append(idx) # Track which rows we actually used

            if not b_emb1:
                continue

            b_emb1 = torch.stack(b_emb1).to(device)
            b_emb2 = torch.stack(b_emb2).to(device)
            
            outputs = seq_model(b_emb1, b_emb2)
            seq_preds.extend(outputs.cpu().numpy().flatten().tolist())
            labels.extend(b_labels)
            
    # Graph Model Inference
    print("Running Graph Model Inference...")
    # We need to construct edge_label_index for the validation set
    # STRICTLY matching the sequence model's predictions (same order/subset)
    # The sequence model loop filtered out pairs missing embeddings.
    # We should ensure we use the SAME filtered set.
    
    # Actually, simpler approach: Filter val_df to only rows where both proteins are in embeddings & mapping
    # This ensures alignment.
    
    filtered_df = val_df[
        val_df["protein1"].isin(embeddings) & 
        val_df["protein2"].isin(embeddings) &
        val_df["protein1"].isin(node_mapping) &
        val_df["protein2"].isin(node_mapping)
    ].copy()
    
    # Re-run sequence inference on filtered_df to be sure (or just use the list if it matches)
    # Let's re-collect properly aligned data.
    
    final_seq_preds = []
    final_graph_preds = []
    final_labels = []
    
    # Prepare batch data for Sequence Model
    batch_emb1 = []
    batch_emb2 = []
    
    # Prepare indices for Graph Model
    g_src = []
    g_dst = []
    
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Aligning Data"):
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        
        # Seq components
        batch_emb1.append(embeddings[p1])
        batch_emb2.append(embeddings[p2])
        
        # Graph components
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        
        final_labels.append(label)

    # Run Sequence Model
    print("Predicting with Sequence Model...")
    # Process in batches to avoid OOM
    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)
    
    final_seq_preds = []
    with torch.no_grad():
        for i in range(0, len(batch_emb1), batch_size):
            e1 = batch_emb1[i:i+batch_size].to(device)
            e2 = batch_emb2[i:i+batch_size].to(device)
            out = seq_model(e1, e2)
            final_seq_preds.extend(out.cpu().numpy().flatten())
            
    # Run Graph Model
    print("Predicting with Graph Model...")
    # Process in batches? GAT supports full edge_label_index.
    # If too large, split.
    
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    
    final_graph_preds = []
    with torch.no_grad():
        # Depending on implementation, g_edge_label_index can be passed whole or chunked
        # For validation set of ~size of train/5, it might fit.
        # But let's chunk to be safe.
        chunk_size = 10000
        for i in range(0, g_edge_label_index.size(1), chunk_size):
            chunk = g_edge_label_index[:, i:i+chunk_size]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            final_graph_preds.extend(out.cpu().numpy().flatten())

    val_labels_np = np.array(final_labels)
    seq_preds_np = np.array(final_seq_preds)
    graph_preds_np = np.array(final_graph_preds)
    
    if len(val_labels_np) == 0:
        print("No valid validation samples found.")
        return

    # 3. Train Ensemble
    ensemble = PPIEnsemble()
    ensemble.train_stacking(seq_preds_np, graph_preds_np, val_labels_np)
    
    # Save
    out_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    ensemble.save(out_path)
    print("Ensemble training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Optional args
    args = parser.parse_args()
    
    # Define default paths
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    graph_data = PROCESSED_DATA_DIR / "ppi_graph.pt"
    
    train_ensemble(seq_path, graph_path, graph_data)
