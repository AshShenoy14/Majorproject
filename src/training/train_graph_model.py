import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import sys
import pandas as pd
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def train(epochs: int = 10, lr: float = 1e-3, graph_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Graph Model on {device}...")
    
    if graph_path and os.path.exists(graph_path):
        # We expect a Data object.
        # However, our process saves the graph structure.
        # For link prediction, we often split edges.
        # But here we have pre-defined train/val/test splits in CSVs.
        # We need to adapt the graph training to these splits.
        # Strategy:
        # 1. Base Graph: Constructed from ALL training positive edges.
        #    (The model message passes on known training edges)
        # 2. Prediction Pairs:
        #    - Train step: Predict on training pairs (pos & neg)
        #    - Val step: Predict on val pairs (pos & neg)
        
        # Load the graph structure (Nodes + Postive Training Edges)
        data = torch.load(graph_path, weights_only=False)
        data = data.to(device)
        
    else:
        print("Graph file not found.")
        return

    # Load Splits
    # We need to map the proteins in CSVs to the node indices in 'data'.
    # This requires the 'node_mapping' to be saved alongside the graph.
    # Assuming 'graph_path' contains (data, mapping) or just data and we reload mapping?
    # Better: modify graph_construction to save mapping too.
    # For now, let's assume we can re-derive or load mapping.
    
    print("Loading datasets for link prediction...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    
    # We need the node mapping to convert protein IDs in DF to node indices
    # Let's assume a mapping file exists or we pass it.
    # For this script template, I will assume a mapping dictionary is available via torch.load
    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if os.path.exists(mapping_path):
        node_mapping = torch.load(mapping_path, weights_only=False)
    else:
        print("Node mapping not found. Cannot align CSVs to Graph.")
        return

    # Helper to get edge indices from DF
    def get_edge_label_index(df):
        src = []
        dst = []
        labels = []
        for _, row in df.iterrows():
            if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
                src.append(node_mapping[row["protein1"]])
                dst.append(node_mapping[row["protein2"]])
                labels.append(row["label"])
        return torch.tensor([src, dst], dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

    train_edge_label_index, train_labels = get_edge_label_index(train_df)
    val_edge_label_index, val_labels = get_edge_label_index(val_df)
    
    train_edge_label_index, train_labels = train_edge_label_index.to(device), train_labels.to(device)
    val_edge_label_index, val_labels = val_edge_label_index.to(device), val_labels.to(device)

    # Model
    model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=64, heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_acc = 0.0

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on the whole graph (message passing on training edges)
        # Predict on training pairs
        # batching might be needed for large datasets (NeighborLoader), but for now full batch
        
        # Note: data.edge_index should be the training graph edges.
        
        outputs = model(data.x, data.edge_index, train_edge_label_index)
        loss = criterion(outputs.squeeze(), train_labels)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(data.x, data.edge_index, val_edge_label_index)
            val_preds = (val_outputs.squeeze() > 0.5).float()
            val_acc = (val_preds == val_labels).sum().item() / val_labels.size(0)
            
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "graph_model_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100) # GAT needs more epochs usually
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--graph_path", type=str, required=True, help="Path to PyG Data object (.pt)")
    args = parser.parse_args()
    
    train(args.epochs, args.lr, args.graph_path)
