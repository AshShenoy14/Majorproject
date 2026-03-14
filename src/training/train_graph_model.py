import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR

CHECKPOINT_VERSION = "v2_encode_once"


def get_model_config(vram_gb: float) -> dict:
    if vram_gb >= 8.0:
        return {"hidden_channels": 128, "heads": 8, "dropout": 0.3, "profile": "large"}
    elif vram_gb >= 6.0:
        return {"hidden_channels": 96, "heads": 4, "dropout": 0.3, "profile": "medium"}
    else:
        return {"hidden_channels": 64, "heads": 4, "dropout": 0.3, "profile": "small"}


def drop_edge(edge_index: torch.Tensor, drop_rate: float = 0.4) -> torch.Tensor:
    """Randomly drop edges from the message-passing graph for regularization."""
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, mask]


def build_positive_edge_index(train_df, node_mapping, device):
    """Build message-passing graph from ONLY positive training interactions."""
    pos_df = train_df[train_df["label"] == 1]
    mp_src, mp_dst = [], []
    for _, row in pos_df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        if p1 in node_mapping and p2 in node_mapping:
            u, v = node_mapping[p1], node_mapping[p2]
            mp_src.extend([u, v])
            mp_dst.extend([v, u])
    return torch.tensor([mp_src, mp_dst], dtype=torch.long, device=device)


def train(epochs: int = 100, lr: float = 0.001, graph_path: str = None,
          edge_batch_size: int = 16384):
    """
    Train GAT Link Predictor.

    Key design decisions:
    1. Graph cleaned to positive-only edges (old pipeline included negatives)
    2. Encode ALL nodes ONCE per epoch (not 40x per batch)
    3. DropEdge prevents trivial detection of direct edges
    4. Gradient accumulation: classifier gradients accumulated across batches,
       then back-propagated through encoder in one pass
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GPU config ───────────────────────────────────────────────────────
    if device.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name} ({total_mem_gb:.1f} GB)")
        config = get_model_config(total_mem_gb)
        if total_mem_gb < 4.5:
            edge_batch_size = 8192
        elif total_mem_gb < 6.0:
            edge_batch_size = 16384
        else:
            edge_batch_size = 65536
    else:
        total_mem_gb = 0
        config = get_model_config(0)
        print("Training on CPU")

    hidden_channels = config["hidden_channels"]
    heads = config["heads"]
    dropout = config["dropout"]

    print(f"Model config [{config['profile']}]: hidden={hidden_channels}, heads={heads}")
    print(f"Training Graph Model on {device}...")

    # ── Load graph ───────────────────────────────────────────────────────
    if not (graph_path and os.path.exists(graph_path)):
        print("Graph file not found.")
        return

    data = torch.load(graph_path, weights_only=False)
    original_edges = data.num_edges
    print(f"Loaded graph: {data.num_nodes} nodes, {original_edges} edges")

    # ── Load splits and mapping ──────────────────────────────────────────
    print("Loading datasets...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if not os.path.exists(mapping_path):
        print("Node mapping not found.")
        return
    node_mapping = torch.load(mapping_path, weights_only=False)

    # ── Rebuild graph: positive-only edges ───────────────────────────────
    clean_edge_index = build_positive_edge_index(train_df, node_mapping, "cpu")
    data.edge_index = clean_edge_index

    if clean_edge_index.size(1) != original_edges:
        print(f"  Cleaned graph: {clean_edge_index.size(1)} edges "
              f"(removed {original_edges - clean_edge_index.size(1)} negative edges)")
        torch.save(data, graph_path)
        print(f"  Saved cleaned graph → {graph_path}")
    else:
        print(f"  Graph already clean: {clean_edge_index.size(1)} edges")

    data = data.to(device)

    # ── Build supervision edge indices ───────────────────────────────────
    def get_edge_label_index(df):
        src, dst, labels = [], [], []
        for _, row in df.iterrows():
            if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
                src.append(node_mapping[row["protein1"]])
                dst.append(node_mapping[row["protein2"]])
                labels.append(row["label"])
        return (torch.tensor([src, dst], dtype=torch.long),
                torch.tensor(labels, dtype=torch.float32))

    train_eli, train_labels = get_edge_label_index(train_df)
    val_eli, val_labels = get_edge_label_index(val_df)

    train_eli = train_eli.to(device)
    train_labels = train_labels.to(device)
    val_eli = val_eli.to(device)
    val_labels = val_labels.to(device)

    num_train = train_labels.size(0)
    num_val = val_labels.size(0)
    num_train_batches = (num_train + edge_batch_size - 1) // edge_batch_size
    num_val_batches = (num_val + edge_batch_size - 1) // edge_batch_size

    print(f"Supervision: {num_train} train / {num_val} val edges")
    print(f"Edge batch size: {edge_batch_size} → {num_train_batches} train batches")

    # ── Model ────────────────────────────────────────────────────────────
    model = GATLinkPredictor(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ── Save config ──────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        "hidden_channels": hidden_channels,
        "heads": heads,
        "dropout": dropout,
        "in_channels": data.x.shape[1],
        "profile": config["profile"],
    }, MODELS_DIR / "graph_model_config.pt")

    # ── Checkpoint ───────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "graph_checkpoint.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    patience = 15
    epochs_no_improve = 0

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("version") != CHECKPOINT_VERSION:
            print("Old checkpoint detected (incompatible procedure). Starting fresh.")
        else:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                epochs_no_improve = ckpt.get("epochs_no_improve", 0)
                print(f"Resumed at epoch {start_epoch}/{epochs} | "
                      f"Best: {best_val_loss:.4f}")
            except RuntimeError:
                print("Checkpoint weights incompatible. Starting fresh.")
    else:
        print("No checkpoint — starting fresh.")

    if start_epoch >= epochs:
        print("Training complete.")
        return

    best_model_path = MODELS_DIR / "graph_model_best.pth"
    DROP_RATE = 0.4

    # ── Training Loop ────────────────────────────────────────────────────
    print(f"\nStarting training (DropEdge={DROP_RATE}, LR={lr})...\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        # 1. DropEdge: randomly mask edges for regularization
        dropped_ei = drop_edge(data.edge_index, drop_rate=DROP_RATE)

        # 2. Encode ALL nodes ONCE per epoch (with gradients tracked)
        z = model.encode(data.x, dropped_ei)
        z_clf = z.detach().requires_grad_(True)

        # 3. Shuffle training edges
        perm = torch.randperm(num_train, device=device)
        eli_shuf = train_eli[:, perm]
        lbl_shuf = train_labels[perm]

        # 4. Score edges in batches, accumulate classifier gradients
        total_loss = 0.0
        pbar = tqdm(
            range(num_train_batches),
            desc=f"Epoch [{epoch + 1}/{epochs}]",
            ncols=100, leave=True,
        )

        for batch_idx in pbar:
            s = batch_idx * edge_batch_size
            e = min(s + edge_batch_size, num_train)

            out = model.decode(z_clf, eli_shuf[:, s:e])
            # Scale loss for correct gradient magnitude with accumulation
            loss = criterion(out.squeeze(), lbl_shuf[s:e]) / num_train_batches
            loss.backward()

            total_loss += loss.item() * num_train_batches  # Unscale for logging
            pbar.set_postfix(loss=f"{loss.item() * num_train_batches:.4f}")

        # 5. Back-propagate accumulated gradients through encoder
        if z_clf.grad is not None:
            z.backward(z_clf.grad)

        # 6. Clip and step once per epoch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_train_loss = total_loss / num_train_batches

        # ── Validation (full graph, no DropEdge) ─────────────────────────
        model.eval()
        with torch.no_grad():
            z_val = model.encode(data.x, data.edge_index)
            total_val_loss = 0.0
            val_correct = 0

            for batch_idx in range(num_val_batches):
                s = batch_idx * edge_batch_size
                e = min(s + edge_batch_size, num_val)

                out = model.decode(z_val, val_eli[:, s:e])
                v_loss = criterion(out.squeeze(), val_labels[s:e])
                total_val_loss += v_loss.item()

                preds = (torch.sigmoid(out.squeeze()) > 0.5).float()
                val_correct += (preds == val_labels[s:e]).sum().item()

        avg_val_loss = total_val_loss / max(num_val_batches, 1)
        val_acc = val_correct / num_val

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Epoch summary ────────────────────────────────────────────────
        print(
            f"\nEpoch [{epoch + 1}/{epochs}] | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Best: {best_val_loss:.4f}"
        )
        print(
            f"  Progress: {epoch + 1}/{epochs} "
            f"({(epoch + 1) / epochs * 100:.0f}%) — "
            f"{epochs - epoch - 1} remaining"
        )

        # ── Save best model ─────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model → {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{patience}")

        # ── Checkpoint ───────────────────────────────────────────────────
        torch.save({
            "version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved")

        # ── Early stopping ───────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n  ✗ Early stopping after {patience} epochs")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--edge_batch_size", type=int, default=16384)
    args = parser.parse_args()

    train(args.epochs, args.lr, args.graph_path, args.edge_batch_size)