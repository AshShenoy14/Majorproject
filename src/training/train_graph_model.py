import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR


def train(epochs: int = 100, lr: float = 0.001, graph_path: str = None, patience: int = 30):
    """
    Train the GAT Link Predictor using full-batch training (Stable Version).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Phase 2/3 GATv2 Model on {device} (Full-Batch)...")

    # ── Load Graph Data ──────────────────────────────────────────────────
    if graph_path and os.path.exists(graph_path):
        data = torch.load(graph_path, weights_only=False)
        data = data.to(device)
    else:
        print("Graph file not found.")
        return

    # ── Load Splits & Node Mapping ───────────────────────────────────────
    print("Loading datasets for link prediction...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df   = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if os.path.exists(mapping_path):
        node_mapping = torch.load(mapping_path, weights_only=False)
    else:
        print("Node mapping not found.")
        return

    def get_edge_label_index(df):
        src, dst, labels = [], [], []
        for _, row in df.iterrows():
            if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
                src.append(node_mapping[row["protein1"]])
                dst.append(node_mapping[row["protein2"]])
                labels.append(row["label"])
        return (torch.tensor([src, dst], dtype=torch.long),
                torch.tensor(labels, dtype=torch.float32))

    train_edge_label_index, train_labels = get_edge_label_index(train_df)
    val_edge_label_index,   val_labels   = get_edge_label_index(val_df)

    train_edge_label_index = train_edge_label_index.to(device)
    train_labels           = train_labels.to(device)
    val_edge_label_index   = val_edge_label_index.to(device)
    val_labels             = val_labels.to(device)

    # ── Model, Optimizer, Loss ───────────────────────────────────────────
    # Optimized architecture (3-layer + skip) with stable dimensions (64/4)
    model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=64, heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Using CosineAnnealingLR for better convergence on CPU
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ── Resume from checkpoint if it exists ──────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / "graph_checkpoint.pt"
    start_epoch     = 0
    best_val_loss   = float("inf")
    epochs_no_improve = 0

    if checkpoint_path.exists():
        print(f"Checking for compatible checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch   = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            print(f"  Resumed at epoch {start_epoch}/{epochs} | Best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Starting fresh: {e}")
    else:
        print("No checkpoint found - starting fresh training.")

    # ── Training Loop ────────────────────────────────────────────────────
    best_model_path = MODELS_DIR / "graph_model_best.pth"
    print("Starting training loop (Stable Full-Batch with Optimized Architecture)...")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        
        # Link prediction on full graph structure
        outputs = model(data.x, data.edge_index, train_edge_label_index)
        loss    = criterion(outputs.squeeze(), train_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss = loss.item()

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            val_outputs = model(data.x, data.edge_index, val_edge_label_index)
            val_loss    = criterion(val_outputs.squeeze(), val_labels).item()
            
            val_probs   = torch.sigmoid(val_outputs.squeeze()).cpu().numpy()
            y_true      = val_labels.cpu().numpy()
            val_preds   = (val_probs > 0.5).astype(float)
            
            val_acc     = (val_preds == y_true).mean()
            val_f1      = f1_score(y_true, val_preds, zero_division=0)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"  [OK] New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--graph_path", type=str, required=True)
    args = parser.parse_args()

    train(args.epochs, args.lr, args.graph_path, args.patience)
