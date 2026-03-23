import os
import sys
# Set project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Set environment variables for E: drive caching before any torch/hf imports
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR
from src.models.graph_model import GATLinkPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
import gc


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
            # Parallel work already started, cannot change threads
            pass
            
        print(f"Runtime: CPU | torch threads={torch.get_num_threads()} | inter-op={torch.get_num_interop_threads()}")
    else:
        # Optimized for memory stability on high-edge-count graphs
        torch.set_num_threads(4)
        print(f"Runtime: CUDA (Threads set to 4 for memory stability)")

    return device


def cooldown_if_needed(seconds: float):
    if seconds and seconds > 0:
        time.sleep(seconds)

def chunked_decode(model, z, edge_label_index, chunk_size=50000):
    """Predict links in chunks to prevent RAM/Drive overflow."""
    src, dst = edge_label_index
    num_edges = src.size(0)
    outputs = []
    
    for i in range(0, num_edges, chunk_size):
        s_chunk = src[i:i+chunk_size]
        d_chunk = dst[i:i+chunk_size]
        out_chunk = model.decode(z, s_chunk, d_chunk)
        outputs.append(out_chunk)
        
    return torch.cat(outputs, dim=0)


def train(
    epochs: int = 100,
    lr: float = 0.001,
    graph_path: str = None,
    patience: int = 30,
    force_cpu: bool = False,
    cpu_threads: int = None,
    edge_batch_size: int = 0,
    cooldown_seconds: float = 0.0,
    model_type: str = "GAT",
    structural_penalty: float = 2.0
):
    """
    Train the GNN Link Predictor using full-batch training (Stable Version).
    model_type: 'GAT' or 'GIN'
    structural_penalty: Multiplier for loss on 'hard' negative samples (if identified).
    """
    device = configure_runtime(force_cpu=force_cpu, cpu_threads=cpu_threads)
    print(f"Training Phase 2/3 {model_type} Model on {device} (Full-Batch)...")

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
        # Support optional 'is_hard' column from future-proofed preprocessor
        is_hard = []
        for _, row in df.iterrows():
            if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
                src.append(node_mapping[row["protein1"]])
                dst.append(node_mapping[row["protein2"]])
                labels.append(row["label"])
                # Heuristic for Phase 2: if label=0 and they appear in 'hard_negatives' (proximal), we weight them more
                # For now, we use a global structural penalty on all negatives if ratio was high
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
    if model_type == "GIN":
        from src.models.graph_model import GINLinkPredictor
        model = GINLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128).to(device)
    else:
        model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128, heads=4).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Custom weighted BCE for Structural Penalty
    def structural_bce_loss(input, target, weights):
        # target=1: Positive, target=0: Negative
        # we want to penalize negatives more if they are 'hard'
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # weights: 1.0 for hard negative, 0.5 for easy negative/positive
        # Apply structural_penalty multiplier specifically to negatives with weight=1.0
        weighted_loss = loss * weights * structural_penalty
        return weighted_loss.mean()

    # Using CosineAnnealingLR for better convergence on CPU
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
            print(f"  Resumed at epoch {start_epoch-1}/{epochs} | Best val loss: {best_val_loss:.4f}")
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
        
        # Force memory cleanup before heavy GNN pass
        gc.collect()

        # Step 1: Encode ALL nodes (GNN Pass)
        # 12k nodes * 512 dimensions is small (~25MB), safe for full-batch
        z = model.encode(data.x, data.edge_index)
        
        # Memory Optimization: Detach z and require grad to isolate decoding chunks 
        # from the GNN computation graph. This prevents OOM (50GB+ spikes).
        z_detached = z.detach().requires_grad_(True)

        # Step 2: Link Prediction (Edge Chunking for Memory Stability)
        src, dst = train_edge_label_index
        num_train_edges = src.size(0)
        chunk_size = 500 # Reduced from 2000 to 500 for extreme memory safety (~500MB per chunk)
        
        epoch_train_loss = 0
        for i in range(0, num_train_edges, chunk_size):
            s_c = src[i:i+chunk_size]
            d_c = dst[i:i+chunk_size]
            lbl_c = train_labels[i:i+chunk_size]
            
            # Predict for chunk using z_detached
            out_c = model.decode(z_detached, s_c, d_c)
            w_c = train_weights[i:i+chunk_size]
            loss_c = structural_bce_loss(out_c.squeeze(), lbl_c, w_c)
            
            # Scale loss by chunk size relative to total edges
            scaled_loss = loss_c * (len(s_c) / num_train_edges)
            
            # Backprop only through decode (accumulates grads in decoder + z_detached)
            scaled_loss.backward()
            epoch_train_loss += loss_c.item() * (len(s_c) / num_train_edges)
            
            # Clear intermediate chunk tensors
            del out_c, loss_c, scaled_loss
            
        # Step 3: Global Backprop through GNN
        # z_detached.grad now contains the sum of all decoding gradients w.r.t embeddings
        z.backward(z_detached.grad)
        
        train_loss = epoch_train_loss

        # Gradient clipping prevents architectural overflow
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Free GNN graph for this epoch
        z = z.detach() 
        gc.collect()

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            gc.collect()
            z_val = model.encode(data.x, data.edge_index)
            
            # Chunked validation to prevent space overflow
            val_outputs = chunked_decode(model, z_val, val_edge_label_index, chunk_size=500)
            val_loss = F.binary_cross_entropy_with_logits(val_outputs.squeeze(), val_labels).item()
            val_probs = torch.sigmoid(val_outputs.squeeze()).cpu().numpy()
            y_true = val_labels.cpu().numpy()




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

        if epochs_no_improve >= 15:
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Maximum PyTorch CPU threads (default: half logical cores)")
    parser.add_argument("--edge-batch-size", type=int, default=0,
                        help="Edge chunk size for train/val. 0 uses full-batch")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0,
                        help="Optional sleep between epochs to reduce thermal load")
    parser.add_argument("--cpu-friendly", action="store_true",
                        help="Enable low-heat CPU preset")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="GAT", choices=["GAT", "GIN"])
    parser.add_argument("--structural-penalty", type=float, default=2.0)
    args = parser.parse_args()

    if args.cpu_friendly:
        args.force_cpu = True
        if args.cpu_threads is None:
            args.cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        if args.edge_batch_size == 0:
            args.edge_batch_size = 0  # 0 indicates Optimized Full-Batch
        if args.cooldown_seconds == 0.0:
            args.cooldown_seconds = 0.25

        print("CPU-friendly preset enabled:")
        print(f"  force_cpu={args.force_cpu} | cpu_threads={args.cpu_threads}")
        print(f"  edge_batch_size={args.edge_batch_size} | cooldown_seconds={args.cooldown_seconds}")

    train(
        epochs=args.epochs,
        lr=args.lr,
        graph_path=args.graph_path,
        patience=args.patience,
        force_cpu=args.force_cpu,
        cpu_threads=args.cpu_threads,
        edge_batch_size=args.edge_batch_size,
        cooldown_seconds=args.cooldown_seconds,
        model_type=args.model_type,
        structural_penalty=args.structural_penalty
    )
