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
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR


def train(epochs: int = 10, lr: float = 1e-3, graph_path: str = None):
    """
    Train the GAT Link Predictor with checkpoint-based resume and best-model saving.

    Checkpoint saved to: checkpoints/graph_checkpoint.pt   (overwritten each epoch)
    Best model saved to: models/graph_model_best.pth       (lowest validation loss)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training Graph Model on {device}...")

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
        print("Node mapping not found. Cannot align CSVs to Graph.")
        return

    # Helper: convert protein-pair CSV rows → edge tensors
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

    print(f"Dataset: {train_labels.size(0)} train / {val_labels.size(0)} val edges")

    # ── Model, Optimizer, Loss ───────────────────────────────────────────
    model     = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=64, heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # ── Resume from checkpoint if it exists ──────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / "graph_checkpoint.pt"
    start_epoch     = 0
    best_val_loss   = float("inf")

    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch   = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}/{epochs} | Best val loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found — starting fresh training.")

    if start_epoch >= epochs:
        print(f"Training already completed ({start_epoch}/{epochs} epochs). Nothing to do.")
        return

    # ── Training Loop ────────────────────────────────────────────────────
    best_model_path = MODELS_DIR / "graph_model_best.pth"

    print("Starting training loop...")
    for epoch in tqdm(range(start_epoch, epochs), desc="Graph Training", ncols=100):
        # --- Train phase (full-batch on graph) ---
        model.train()
        optimizer.zero_grad()

        outputs = model(data.x, data.edge_index, train_edge_label_index)
        loss    = criterion(outputs.squeeze(), train_labels)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            val_outputs = model(data.x, data.edge_index, val_edge_label_index)
            val_loss_t  = criterion(val_outputs.squeeze(), val_labels)
            val_loss    = val_loss_t.item()
            val_preds   = (val_outputs.squeeze() > 0.5).float()
            val_acc     = (val_preds == val_labels).sum().item() / val_labels.size(0)

        # --- Epoch summary ---
        print(
            f"\nEpoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )
        print(f"  Progress: {epoch+1}/{epochs} epochs done "
              f"({(epoch+1)/epochs*100:.0f}%) — {epochs - epoch - 1} remaining")

        # --- Save best model if validation loss improved ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved → {best_model_path}")

        # --- Save checkpoint (overwrite each epoch) ---
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "best_val_loss": best_val_loss,
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved → {checkpoint_path}")

    print(f"\nGraph Model training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--graph_path", type=str, required=True,
                        help="Path to PyG Data object (.pt)")
    args = parser.parse_args()

    train(args.epochs, args.lr, args.graph_path)
