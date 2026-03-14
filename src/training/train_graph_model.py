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

CHECKPOINT_VERSION = "v3_full_graph"


def get_model_config(vram_gb: float) -> dict:
    if vram_gb >= 8.0:
        return {"hidden_channels": 128, "heads": 8, "dropout": 0.3, "profile": "large"}
    elif vram_gb >= 6.0:
        return {"hidden_channels": 96, "heads": 4, "dropout": 0.3, "profile": "medium"}
    else:
        return {"hidden_channels": 64, "heads": 4, "dropout": 0.3, "profile": "small"}


def drop_edge(edge_index: torch.Tensor, drop_rate: float = 0.3) -> torch.Tensor:
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, mask]


def build_complete_positive_graph(node_mapping):
    """
    Build message-passing graph from ALL positive interactions across all splits.
    
    This is standard transductive link prediction:
    - Graph structure includes all known interactions
    - Supervision edges (train/val) are separate
    - Model learns to distinguish real interactions from random pairs
    """
    all_src, all_dst = [], []
    seen = set()

    for csv_name in ["train.csv", "val.csv", "test.csv"]:
        path = PROCESSED_DATA_DIR / csv_name
        if path.exists():
            df = pd.read_csv(path)
            pos_df = df[df["label"] == 1]
            for _, row in pos_df.iterrows():
                p1, p2 = row["protein1"], row["protein2"]
                if p1 in node_mapping and p2 in node_mapping:
                    u, v = node_mapping[p1], node_mapping[p2]
                    pair = (min(u, v), max(u, v))
                    if pair not in seen:
                        seen.add(pair)
                        all_src.extend([u, v])
                        all_dst.extend([v, u])

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    print(f"  Complete graph: {len(seen)} undirected edges "
          f"({edge_index.size(1)} directed)")
    return edge_index


def train(epochs: int = 100, lr: float = 0.001, graph_path: str = None,
          edge_batch_size: int = 16384):
    """
    Train GAT Link Predictor (transductive link prediction).

    Key design:
    1. Message-passing graph uses ALL positive edges from train+val+test
       (standard transductive setup — graph structure is fully known)
    2. Supervision edges split into train/val for classifier training
    3. DropEdge regularization prevents trivial edge detection
    4. Encode ALL nodes ONCE per epoch, batch the edge scoring
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")

    # ── Load mapping ─────────────────────────────────────────────────────
    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if not os.path.exists(mapping_path):
        print("Node mapping not found.")
        return
    node_mapping = torch.load(mapping_path, weights_only=False)

    # ── Rebuild graph with ALL positive edges (transductive) ─────────────
    print("Building complete positive graph from all splits...")
    clean_edge_index = build_complete_positive_graph(node_mapping)

    if clean_edge_index.size(1) != data.num_edges:
        print(f"  Updated: {data.num_edges} -> {clean_edge_index.size(1)} edges")
        data.edge_index = clean_edge_index
        torch.save(data, graph_path)
        print(f"  Saved updated graph -> {graph_path}")
    else:
        print(f"  Graph already correct: {data.num_edges} edges")

    data = data.to(device)

    # ── Load splits ──────────────────────────────────────────────────────
    print("Loading supervision edges...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

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
    print(f"Edge batch size: {edge_batch_size} -> {num_train_batches} train batches")

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
    patience = 10
    epochs_no_improve = 0

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("version") != CHECKPOINT_VERSION:
            print("Old checkpoint (incompatible version). Starting fresh.")
        else:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                epochs_no_improve = ckpt.get("epochs_no_improve", 0)
                print(f"Resumed at epoch {start_epoch}/{epochs} | Best: {best_val_loss:.4f}")
            except RuntimeError:
                print("Checkpoint weights incompatible. Starting fresh.")
    else:
        print("No checkpoint — starting fresh.")

    if start_epoch >= epochs:
        print("Training complete.")
        return

    best_model_path = MODELS_DIR / "graph_model_best.pth"
    DROP_RATE = 0.3

    # ── Training Loop ────────────────────────────────────────────────────
    print(f"\nStarting training (DropEdge={DROP_RATE}, LR={lr})...\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        # 1. DropEdge regularization
        dropped_ei = drop_edge(data.edge_index, drop_rate=DROP_RATE)

        # 2. Encode ALL nodes ONCE
        z = model.encode(data.x, dropped_ei)
        z_clf = z.detach().requires_grad_(True)

        # 3. Shuffle and batch supervision edges
        perm = torch.randperm(num_train, device=device)
        eli_shuf = train_eli[:, perm]
        lbl_shuf = train_labels[perm]

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
            loss = criterion(out.squeeze(), lbl_shuf[s:e]) / num_train_batches
            loss.backward()

            total_loss += loss.item() * num_train_batches
            pbar.set_postfix(loss=f"{loss.item() * num_train_batches:.4f}")

        # 4. Backprop through encoder
        if z_clf.grad is not None:
            z.backward(z_clf.grad)

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model -> {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{patience}")

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

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping after {patience} epochs")
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