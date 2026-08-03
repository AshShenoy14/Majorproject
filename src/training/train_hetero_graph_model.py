import os
import sys
import argparse
import time
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import HeteroData

# Set project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR
from src.models.hetero_graph_model import HeteroGNNLinkPredictor

def configure_runtime(force_cpu: bool = False, cpu_threads: int = None):
    """Configure runtime device and CPU thread counts for thermally stable training."""
    use_cpu = force_cpu or (not torch.cuda.is_available())
    device = torch.device("cpu" if use_cpu else "cuda")

    if device.type == "cpu":
        if cpu_threads is None:
            cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        cpu_threads = max(1, int(cpu_threads))
        
        try:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
        except RuntimeError:
            pass
            
        print(f"Runtime: CPU | torch threads={torch.get_num_threads()} | inter-op={torch.get_num_interop_threads()}")
    else:
        torch.set_num_threads(4)
        print(f"Runtime: CUDA (Threads set to 4 for memory stability)")

    return device

def cooldown_if_needed(seconds: float):
    if seconds and seconds > 0:
        time.sleep(seconds)

def chunked_decode(model, z_protein, edge_label_index, chunk_size=50000):
    """Predict protein-protein links in chunks to prevent RAM overflow."""
    src, dst = edge_label_index
    num_edges = src.size(0)
    outputs = []
    
    for i in range(0, num_edges, chunk_size):
        s_chunk = src[i:i+chunk_size]
        d_chunk = dst[i:i+chunk_size]
        chunk_edge_label_index = torch.stack([s_chunk, d_chunk], dim=0)
        out_chunk = model.decode(z_protein, chunk_edge_label_index)
        outputs.append(out_chunk)
        
    return torch.cat(outputs, dim=0)

def train_hetero(
    epochs: int = 100,
    lr: float = 0.001,
    graph_path: str = None,
    patience: int = 30,
    force_cpu: bool = False,
    cpu_threads: int = None,
    edge_batch_size: int = 0,
    cooldown_seconds: float = 0.0,
    structural_penalty: float = 2.0
):
    device = configure_runtime(force_cpu=force_cpu, cpu_threads=cpu_threads)
    print(f"Training Heterogeneous GNN Model on {device} (Full-Batch)...")

    # ── Load Hetero Graph Data ───────────────────────────────────────────
    if graph_path and os.path.exists(graph_path):
        data = torch.load(graph_path, weights_only=False)
        data = data.to(device)
    else:
        print("Hetero Graph file not found.")
        return

    # ── Load Splits & Node Mapping ───────────────────────────────────────
    print("Loading datasets for link prediction...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df   = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    
    mappings_path = PROCESSED_DATA_DIR / "ppi_hetero_mappings.pt"
    if mappings_path.exists():
        mappings = torch.load(mappings_path, weights_only=False)
        protein_mapping = mappings["protein"]
    else:
        print("Hetero mappings not found.")
        return

    def get_edge_label_index(df):
        src, dst, labels = [], [], []
        is_hard = []
        for _, row in df.iterrows():
            if row["protein1"] in protein_mapping and row["protein2"] in protein_mapping:
                src.append(protein_mapping[row["protein1"]])
                dst.append(protein_mapping[row["protein2"]])
                labels.append(row["label"])
                is_hard.append(1.0 if row.get("is_hard", 0) else 0.5) 
        return (torch.tensor([src, dst], dtype=torch.long),
                torch.tensor(labels, dtype=torch.float32),
                torch.tensor(is_hard, dtype=torch.float32))

    train_edge_label_index, train_labels, train_weights = get_edge_label_index(train_df)
    val_edge_label_index,   val_labels,   _            = get_edge_label_index(val_df)

    train_edge_label_index = train_edge_label_index.to(device)
    train_labels           = train_labels.to(device)
    train_weights          = train_weights.to(device)
    val_edge_label_index   = val_edge_label_index.to(device)
    val_labels             = val_labels.to(device)

    # ── Model, Optimizer, Loss ───────────────────────────────────────────
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    model = HeteroGNNLinkPredictor(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_channels=128
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    def focal_loss(inputs, targets, alpha=0.5, gamma=2.0):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Resume from checkpoint if it exists ──────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / "hetero_graph_checkpoint.pt"
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
            print(f"  Resumed at epoch {start_epoch-1}/{epochs} | Best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Starting fresh: {e}")
    else:
        print("No checkpoint found - starting fresh training.")

    # ── Training Loop ────────────────────────────────────────────────────
    best_model_path = MODELS_DIR / "hetero_graph_model_best.pth"
    print("Starting training loop (Stable Full-Batch with Heterogeneous Neighborhood Context)...")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        gc.collect()

        # Step 1: Encode ALL nodes (Heterogeneous Conv Pass)
        z_dict = model.encode(data.x_dict, data.edge_index_dict)
        
        # Memory Optimization: Detach protein embeddings for chunked decoding
        z_protein_detached = z_dict['protein'].detach().requires_grad_(True)

        # Step 2: Link Prediction on Protein-Protein pairs
        src, dst = train_edge_label_index
        num_train_edges = src.size(0)
        chunk_size = 1000 
        
        epoch_train_loss = 0
        for i in range(0, num_train_edges, chunk_size):
            s_c = src[i:i+chunk_size]
            d_c = dst[i:i+chunk_size]
            lbl_c = train_labels[i:i+chunk_size]
            w_c = train_weights[i:i+chunk_size]
            
            chunk_edge_label_index = torch.stack([s_c, d_c], dim=0)
            out_c = model.decode(z_protein_detached, chunk_edge_label_index)
            
            loss_c = (focal_loss(out_c.squeeze(), lbl_c, alpha=0.3, gamma=2.5) * w_c).mean()
            scaled_loss = loss_c * 10.0 
            scaled_loss.backward()
            epoch_train_loss += loss_c.item() * (len(s_c) / num_train_edges)
            
            del out_c, loss_c, scaled_loss
            
        # Step 3: Global Backprop through GNN using protein embeddings gradient
        z_dict['protein'].backward(z_protein_detached.grad)
        
        train_loss = epoch_train_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Free graph memory
        z_dict = {ntype: z.detach() for ntype, z in z_dict.items()}
        gc.collect()

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            gc.collect()
            z_dict_val = model.encode(data.x_dict, data.edge_index_dict)
            val_outputs = chunked_decode(model, z_dict_val['protein'], val_edge_label_index, chunk_size=1000)
            val_loss = F.binary_cross_entropy_with_logits(val_outputs.squeeze(), val_labels).item()
            val_probs = torch.sigmoid(val_outputs.squeeze()).cpu().numpy()
            y_true = val_labels.cpu().numpy()

            val_preds = (val_probs > 0.5).astype(float)
            val_acc   = (val_preds == y_true).mean()
            val_f1    = f1_score(y_true, val_preds, zero_division=0)

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

        cooldown_if_needed(cooldown_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--cpu-friendly", action="store_true", help="Enable CPU preset")
    parser.add_argument("--graph_path", type=str, default="data/processed/ppi_hetero_graph.pt")
    args = parser.parse_args()

    if args.cpu_friendly:
        args.force_cpu = True
        if args.cpu_threads is None:
            args.cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        if args.cooldown_seconds == 0.0:
            args.cooldown_seconds = 0.1

    train_hetero(
        epochs=args.epochs,
        lr=args.lr,
        graph_path=args.graph_path,
        patience=args.patience,
        force_cpu=args.force_cpu,
        cpu_threads=args.cpu_threads,
        cooldown_seconds=args.cooldown_seconds
    )
