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

def train(epochs: int = 100, lr: float = 0.005, graph_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Graph Model on {device}...")
    
    if graph_path and os.path.exists(graph_path):
        data = torch.load(graph_path, weights_only=False)
        data = data.to(device)
    else:
        print("Graph file not found.")
        return

    # Load Splits
    print("Loading datasets for link prediction...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    
    # Load node mapping
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

    # Model — increased capacity: hidden=128, heads=8
    model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128, heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # BCEWithLogitsLoss for better gradient stability (model outputs raw logits)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass — model returns raw logits
        outputs = model(data.x, data.edge_index, train_edge_label_index)
        loss = criterion(outputs.squeeze(), train_labels)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(data.x, data.edge_index, val_edge_label_index)
            # Apply sigmoid only during evaluation
            val_probs = torch.sigmoid(val_outputs.squeeze())
            val_preds = (val_probs > 0.5).float()
            val_acc = (val_preds == val_labels).sum().item() / val_labels.size(0)
            
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "graph_model_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--graph_path", type=str, required=True, help="Path to PyG Data object (.pt)")
    args = parser.parse_args()
    
    train(args.epochs, args.lr, args.graph_path)
