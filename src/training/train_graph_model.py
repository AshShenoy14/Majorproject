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

CHECKPOINT_VERSION = "v4_no_leakage"


def get_model_config(vram_gb: float) -> dict:
    if vram_gb >= 8.0:
        return {"hidden_channels": 128, "heads": 8,
                "dropout": 0.3, "profile": "large"}
    elif vram_gb >= 6.0:
        return {"hidden_channels": 96, "heads": 4,
                "dropout": 0.3, "profile": "medium"}
    else:
        return {"hidden_channels": 64, "heads": 4,
                "dropout": 0.3, "profile": "small"}


def drop_edge(edge_index: torch.Tensor, drop_rate: float = 0.3):
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, mask]


def build_train_graph(node_mapping):
    """
    Build message-passing graph from TRAIN + VAL positives ONLY.
    
    *** NEVER include test.csv — that is label leakage ***
    
    Why include val: val edges are known interactions used for
    hyperparameter tuning. The supervision signal (what the model
    predicts on) is separate from the graph structure.
    """
    all_src, all_dst = [], []
    seen = set()

    for csv_name in ["train.csv", "val.csv"]:  # ← NO test.csv!
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
    print(f"  Train graph: {len(seen)} undirected edges "
          f"({edge_index.size(1)} directed)")
    print(f"  *** test.csv edges EXCLUDED (no leakage) ***")
    return edge_index


def train_with_fallback_alignment(
    model, data, train_eli, train_labels,
    val_eli, val_labels, device,
    epochs, lr, edge_batch_size, patience=10,
    pos_weight=None
):
    """
    Training loop that also trains the sequence_fallback encoder
    to align with GAT embeddings.
    
    This ensures new proteins (without graph context) get
    similar-quality embeddings at inference time.
    
    Args:
        pos_weight: Weight for positive class in BCEWithLogitsLoss
                    to handle class imbalance.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    alignment_criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    num_train = train_labels.size(0)
    num_val = val_labels.size(0)
    num_train_batches = (num_train + edge_batch_size - 1) // edge_batch_size
    num_val_batches = (num_val + edge_batch_size - 1) // edge_batch_size

    # Checkpoint setup
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "graph_checkpoint.pt"
    best_model_path = MODELS_DIR / "graph_model_best.pth"

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        if ckpt.get("version") == CHECKPOINT_VERSION:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                epochs_no_improve = ckpt.get("epochs_no_improve", 0)
                print(f"Resumed epoch {start_epoch}/{epochs} "
                      f"| Best: {best_val_loss:.4f}")
            except RuntimeError:
                print("Checkpoint incompatible. Starting fresh.")
        else:
            print("Old checkpoint version. Starting fresh.")

    if start_epoch >= epochs:
        print("Training already complete.")
        return

    DROP_RATE = 0.3
    ALIGNMENT_WEIGHT = 0.1  # Weight for fallback alignment loss

    print(f"\nTraining (DropEdge={DROP_RATE}, LR={lr}, "
          f"Alignment={ALIGNMENT_WEIGHT})...\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        # 1. DropEdge
        dropped_ei = drop_edge(data.edge_index, drop_rate=DROP_RATE)

        # 2. Encode all nodes with GAT
        z_gat = model.encode(data.x, dropped_ei)

        # 3. Encode all nodes with fallback (no graph)
        z_fallback = model.encode_sequences(data.x)

        # 4. Alignment loss: fallback should match GAT output
        alignment_loss = alignment_criterion(z_fallback, z_gat.detach())

        # 5. Edge prediction loss (using GAT embeddings)
        z_clf = z_gat.detach().requires_grad_(True)

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
            loss = (criterion(out.squeeze(), lbl_shuf[s:e])
                    / num_train_batches)
            loss.backward()

            total_loss += loss.item() * num_train_batches
            pbar.set_postfix(loss=f"{loss.item() * num_train_batches:.4f}")

        # 6. Backprop through encoder
        if z_clf.grad is not None:
            z_gat.backward(z_clf.grad)

        # 7. Add alignment loss
        (alignment_loss * ALIGNMENT_WEIGHT).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_train_loss = total_loss / num_train_batches

        # Validation
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
            f"Align: {alignment_loss.item():.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved")
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

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


def train(epochs=100, lr=0.001, graph_path=None, edge_batch_size=16384):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
        total_mem_gb = total_mem / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_mem_gb:.1f} GB)")
        config = get_model_config(total_mem_gb)
        if total_mem_gb < 4.5:
            edge_batch_size = 8192
        elif total_mem_gb < 6.0:
            edge_batch_size = 16384
        else:
            edge_batch_size = 65536
    else:
        config = get_model_config(0)

    print(f"Config [{config['profile']}]: "
          f"hidden={config['hidden_channels']}, heads={config['heads']}")

    # Load graph
    if not (graph_path and os.path.exists(graph_path)):
        print("Graph file not found.")
        return

    data = torch.load(graph_path, weights_only=False)
    print(f"Loaded: {data.num_nodes} nodes, {data.num_edges} edges")

    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if not os.path.exists(mapping_path):
        print("Node mapping not found.")
        return
    node_mapping = torch.load(mapping_path, weights_only=False)

    # *** CRITICAL FIX: Build graph WITHOUT test edges ***
    print("Building graph (train+val only, NO test leakage)...")
    clean_edge_index = build_train_graph(node_mapping)
    data.edge_index = clean_edge_index

    # Save clean graph
    torch.save(data, graph_path)
    print(f"  Saved clean graph → {graph_path}")

    data = data.to(device)

    # Load supervision edges
    print("Loading supervision edges...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    def get_edge_label_index(df):
        src, dst, labels = [], [], []
        for _, row in df.iterrows():
            p1, p2 = row["protein1"], row["protein2"]
            if p1 in node_mapping and p2 in node_mapping:
                src.append(node_mapping[p1])
                dst.append(node_mapping[p2])
                labels.append(row["label"])
        return (torch.tensor([src, dst], dtype=torch.long),
                torch.tensor(labels, dtype=torch.float32))

    train_eli, train_labels = get_edge_label_index(train_df)
    val_eli, val_labels = get_edge_label_index(val_df)

    train_eli = train_eli.to(device)
    train_labels = train_labels.to(device)
    val_eli = val_eli.to(device)
    val_labels = val_labels.to(device)

    print(f"Supervision: {train_labels.size(0)} train / "
          f"{val_labels.size(0)} val")

    # Compute class weights for imbalanced data
    n_pos = (train_labels == 1).sum().item()
    n_neg = (train_labels == 0).sum().item()
    if n_pos > 0 and n_neg > 0:
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        print(f"Class balance: {int(n_pos)} pos / {int(n_neg)} neg → pos_weight={pw.item():.3f}")
    else:
        pw = torch.tensor([1.0], dtype=torch.float32).to(device)
        print(f"⚠ Single-class data — pos_weight=1.0")

    # Model
    model = GATLinkPredictor(
        in_channels=data.x.shape[1],
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Save config
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        "hidden_channels": config["hidden_channels"],
        "heads": config["heads"],
        "dropout": config["dropout"],
        "in_channels": data.x.shape[1],
        "profile": config["profile"],
    }, MODELS_DIR / "graph_model_config.pt")

    # Train
    train_with_fallback_alignment(
        model, data, train_eli, train_labels,
        val_eli, val_labels, device,
        epochs, lr, edge_batch_size,
        pos_weight=pw,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--edge_batch_size", type=int, default=16384)
    args = parser.parse_args()
    train(args.epochs, args.lr, args.graph_path, args.edge_batch_size)