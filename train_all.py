"""
train_all.py — Complete training pipeline for TransGraph-PPI
Trains all 3 models from scratch and evaluates on test set.
Does NOT run run_pipeline.py. Uses pre-computed embeddings and graph.

Usage:
    python train_all.py
    python train_all.py --fresh          # Delete old checkpoints for a clean start
    python train_all.py --seq-epochs 50  # Override sequence model epochs
    python train_all.py --gat-epochs 100 # Override graph model epochs
"""

import sys
import os
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.utils.dataset import PPIDataset
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR
from src.evaluation.metrics_reporter import report_all_metrics


def clear_checkpoints():
    """Delete old checkpoints so training starts fresh."""
    for name in ["sequence_checkpoint.pt", "graph_checkpoint.pt"]:
        path = CHECKPOINT_DIR / name
        if path.exists():
            os.remove(path)
            print(f"  Deleted checkpoint: {path}")
    print("  Checkpoints cleared.\n")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1: Train Sequence Model (ESM-MLP)
# ════════════════════════════════════════════════════════════════════════════
def train_sequence_model(epochs=50, batch_size=64, lr=1e-3):
    print("\n" + "=" * 70)
    print("  PHASE 1/3 — Training Sequence Model (ESM-MLP)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load embeddings
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    if not emb_path.exists():
        print("ERROR: embeddings.pt not found. Run run_pipeline.py first to extract embeddings.")
        return
    
    print("  Loading embeddings (this may take a moment for 13GB file)...")
    embeddings = torch.load(emb_path, weights_only=False)
    print(f"  Loaded {len(embeddings)} protein embeddings.")

    # Datasets
    train_dataset = PPIDataset(PROCESSED_DATA_DIR / "train.csv", embeddings)
    val_dataset = PPIDataset(PROCESSED_DATA_DIR / "val.csv", embeddings)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  Dataset: {len(train_dataset)} train / {len(val_dataset)} val")
    print(f"  Batch size: {batch_size} | Train batches/epoch: {len(train_loader)}")

    # Model
    input_dim = 320  # ESM-2 t6_8M_UR50D
    model = SequencePPIModel(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Resume from checkpoint
    checkpoint_path = CHECKPOINT_DIR / "sequence_checkpoint.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    patience = 7
    epochs_no_improve = 0

    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        # Reconstruct scheduler state
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}/{epochs} | Best val loss: {best_val_loss:.4f}")
    else:
        print("  Starting fresh training.")

    if start_epoch >= epochs:
        print(f"  Training already completed ({start_epoch}/{epochs}). Skipping.")
        del embeddings
        gc.collect()
        return

    # Training loop
    best_model_path = MODELS_DIR / "sequence_model_best.pth"

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"  Epoch [{epoch+1}/{epochs}] Train", leave=True, ncols=100)
        for emb1, emb2, labels in pbar:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(emb1, emb2)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
                outputs = model(emb1, emb2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Best: {best_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✓ New best model saved → {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"    No improvement for {epochs_no_improve}/{patience} epochs.")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }, checkpoint_path)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n    ✗ Early stopping at epoch {epoch+1}.")
            break

    print(f"\n  Sequence Model training complete. Best val loss: {best_val_loss:.4f}")

    # Free memory
    del embeddings, train_dataset, val_dataset, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
#  Phase 2: Train Graph Model (GATv2)
# ════════════════════════════════════════════════════════════════════════════
def train_graph_model(epochs=100, lr=0.001, patience=30):
    print("\n" + "=" * 70)
    print("  PHASE 2/3 — Training Graph Model (GATv2)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load graph
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    if not graph_path.exists():
        print("ERROR: ppi_graph.pt not found.")
        return

    data = torch.load(graph_path, weights_only=False).to(device)
    
    # --- NEW: Topological Feature Injection ---
    # Calculate degree for each node to give the model structural context
    from torch_geometric.utils import degree
    print("  Injecting Topological Features (Node Degree)...")
    deg = degree(data.edge_index[0], data.x.shape[0]).view(-1, 1)
    # Normalize degree
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-6)
    data.x = torch.cat([data.x, deg_norm], dim=-1)
    
    print(f"  Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges, {data.x.shape[1]} features")

    # Load splits & node mapping
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"

    if not mapping_path.exists():
        print("ERROR: ppi_graph_mapping.pt not found.")
        return

    node_mapping = torch.load(mapping_path, weights_only=False)

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
    val_edge_label_index, val_labels = get_edge_label_index(val_df)

    train_edge_label_index = train_edge_label_index.to(device)
    train_labels = train_labels.to(device)
    val_edge_label_index = val_edge_label_index.to(device)
    val_labels = val_labels.to(device)

    # Model: Increased to 128 channels for stability
    model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128, heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    
    # More conservative OneCycle config
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 0.5, total_steps=epochs, 
        pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    # Resume from checkpoint
    checkpoint_path = CHECKPOINT_DIR / "graph_checkpoint.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if checkpoint_path.exists():
        print(f"  Checking for compatible checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  Resumed at epoch {start_epoch}/{epochs} | Best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"  Starting fresh: {e}")
    else:
        print("  No checkpoint found - starting fresh.")

    if start_epoch >= epochs:
        print(f"  Training already completed ({start_epoch}/{epochs}). Skipping.")
        return

    # Training loop
    best_model_path = MODELS_DIR / "graph_model_best.pth"
    print("  Starting training loop (Full-Batch GATv2)...")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(data.x, data.edge_index, train_edge_label_index)
        loss = criterion(outputs.squeeze(), train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(data.x, data.edge_index, val_edge_label_index)
            val_loss = criterion(val_outputs.squeeze(), val_labels).item()
            val_probs = torch.sigmoid(val_outputs.squeeze()).cpu().numpy()
            y_true = val_labels.cpu().numpy()
            val_preds = (val_probs > 0.5).astype(float)
            val_acc = (val_preds == y_true).mean()
            val_f1 = f1_score(y_true, val_preds, zero_division=0)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    [OK] New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }, checkpoint_path)

    print(f"\n  Graph Model training complete. Best val loss: {best_val_loss:.4f}")

    del model, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
#  Phase 3: Train Ensemble (XGBoost Stacking)
# ════════════════════════════════════════════════════════════════════════════
def train_ensemble_model():
    print("\n" + "=" * 70)
    print("  PHASE 3/3 — Training Ensemble Meta-Learner (XGBoost)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load base models
    seq_model_path = MODELS_DIR / "sequence_model_best.pth"
    graph_model_path = MODELS_DIR / "graph_model_best.pth"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    if not seq_model_path.exists():
        print("ERROR: sequence_model_best.pth not found. Train sequence model first.")
        return
    if not graph_model_path.exists():
        print("ERROR: graph_model_best.pth not found. Train graph model first.")
        return

    # Sequence Model
    input_dim = 320
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
    seq_model.eval()
    print(f"  Loaded Sequence Model from {seq_model_path}")

    # Graph Model
    graph_data = torch.load(graph_data_path, weights_only=False).to(device)
    
    # Inject topology (must match training Phase 2)
    from torch_geometric.utils import degree
    deg = degree(graph_data.edge_index[0], graph_data.x.shape[0]).view(-1, 1)
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-6)
    graph_data.x = torch.cat([graph_data.x, deg_norm], dim=-1)

    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128, heads=4).to(device)
    graph_model.load_state_dict(torch.load(graph_model_path, map_location=device))
    graph_model.eval()
    print(f"  Loaded Hybrid Graph Model from {graph_model_path}")

    # Load support files
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"

    print("  Loading embeddings...")
    embeddings = torch.load(emb_path, weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)

    # Generate predictions on validation set
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    filtered_df = val_df[
        val_df["protein1"].isin(embeddings) &
        val_df["protein2"].isin(embeddings) &
        val_df["protein1"].isin(node_mapping) &
        val_df["protein2"].isin(node_mapping)
    ].copy()

    print(f"  Generating predictions on {len(filtered_df)} validation samples...")

    # --- NEW: Biological Coincidence Features ---
    print("  Extracting Biological Features (Localization)...")
    bio_path = PROCESSED_DATA_DIR / "bio_metadata_cache.csv"
    bio_df = pd.read_csv(bio_path).set_index("protein_id")["localization"].to_dict() if bio_path.exists() else {}

    batch_emb1, batch_emb2 = [], []
    g_src, g_dst = [], []
    final_labels = []
    val_bio_features = []

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="  Aligning Data"):
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        e1 = embeddings[p1]
        e2 = embeddings[p2]
        batch_emb1.append(e1.mean(dim=0) if e1.dim() > 1 else e1)
        batch_emb2.append(e2.mean(dim=0) if e2.dim() > 1 else e2)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        final_labels.append(label)

        # 1 if same localization, 0 otherwise
        loc1, loc2 = bio_df.get(p1, "unk1"), bio_df.get(p2, "unk2")
        val_bio_features.append([1.0 if loc1 == loc2 and loc1 != "unk1" else 0.0])

    val_bio_features = np.array(val_bio_features)

    # Sequence predictions
    print("  Predicting with Sequence Model...")
    batch_emb1_t = torch.stack(batch_emb1)
    batch_emb2_t = torch.stack(batch_emb2)
    batch_size = 32

    final_seq_preds = []
    with torch.no_grad():
        for i in range(0, len(batch_emb1_t), batch_size):
            e1 = batch_emb1_t[i:i + batch_size].to(device)
            e2 = batch_emb2_t[i:i + batch_size].to(device)
            out = seq_model(e1, e2)
            probs = torch.sigmoid(out)
            final_seq_preds.extend(probs.cpu().numpy().flatten())

    # Graph predictions
    print("  Predicting with Graph Model...")
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    final_graph_preds = []
    with torch.no_grad():
        chunk_size = 10000
        for i in range(0, g_edge_label_index.size(1), chunk_size):
            chunk = g_edge_label_index[:, i:i + chunk_size]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            probs = torch.sigmoid(out)
            final_graph_preds.extend(probs.cpu().numpy().flatten())

    val_labels_np = np.array(final_labels)
    seq_preds_np = np.array(final_seq_preds)
    graph_preds_np = np.array(final_graph_preds)

    # Train ensemble with bio features
    ensemble = PPIEnsemble()
    ensemble.train_stacking(seq_preds_np, graph_preds_np, val_labels_np, bio_features=val_bio_features)

    out_path = MODELS_DIR / "ensemble_model.pkl"
    ensemble.save(out_path)
    print("  Ensemble training complete.")

    # Free memory before evaluation
    del embeddings
    gc.collect()

    return seq_model, graph_model, graph_data, node_mapping, ensemble


# ════════════════════════════════════════════════════════════════════════════
#  Phase 4: Evaluate All Models on Test Set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_all():
    print("\n" + "=" * 70)
    print("  EVALUATION — All Models on Test Set")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    seq_model_path = MODELS_DIR / "sequence_model_best.pth"
    graph_model_path = MODELS_DIR / "graph_model_best.pth"
    ensemble_path = MODELS_DIR / "ensemble_model.pkl"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    # Sequence Model
    input_dim = 320
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
    seq_model.eval()

    # Graph Model
    graph_data = torch.load(graph_data_path, weights_only=False).to(device)
    
    # Inject topology for evaluation
    from torch_geometric.utils import degree
    deg = degree(graph_data.edge_index[0], graph_data.x.shape[0]).view(-1, 1)
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-6)
    graph_data.x = torch.cat([graph_data.x, deg_norm], dim=-1)

    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128, heads=4).to(device)
    graph_model.load_state_dict(torch.load(graph_model_path, map_location=device))
    graph_model.eval()

    # Ensemble
    ensemble = PPIEnsemble(str(ensemble_path))

    # Load support files
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"

    print("  Loading embeddings for evaluation...")
    embeddings = torch.load(emb_path, weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)

    # Use TEST set for final evaluation
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    filtered_df = test_df[
        test_df["protein1"].isin(embeddings) &
        test_df["protein2"].isin(embeddings) &
        test_df["protein1"].isin(node_mapping) &
        test_df["protein2"].isin(node_mapping)
    ].copy()

    print(f"  Evaluating on {len(filtered_df)} test samples...")

    # Bio Features for Test
    print("  Extracting Bio Features for Test...")
    bio_path = PROCESSED_DATA_DIR / "bio_metadata_cache.csv"
    bio_df = pd.read_csv(bio_path).set_index("protein_id")["localization"].to_dict() if bio_path.exists() else {}
    test_bio_features = []

    batch_emb1, batch_emb2 = [], []
    g_src, g_dst = [], []
    final_labels = []

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="  Preparing test data"):
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        e1 = embeddings[p1]
        e2 = embeddings[p2]
        batch_emb1.append(e1.mean(dim=0) if e1.dim() > 1 else e1)
        batch_emb2.append(e2.mean(dim=0) if e2.dim() > 1 else e2)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        final_labels.append(label)
        
        # Localization match
        loc1, loc2 = bio_df.get(p1, "unk1"), bio_df.get(p2, "unk2")
        test_bio_features.append([1.0 if loc1 == loc2 and loc1 != "unk1" else 0.0])

    test_bio_features = np.array(test_bio_features)

    # Sequence predictions on test
    batch_emb1_t = torch.stack(batch_emb1)
    batch_emb2_t = torch.stack(batch_emb2)
    batch_size = 32

    seq_preds = []
    with torch.no_grad():
        for i in range(0, len(batch_emb1_t), batch_size):
            e1 = batch_emb1_t[i:i + batch_size].to(device)
            e2 = batch_emb2_t[i:i + batch_size].to(device)
            out = seq_model(e1, e2)
            probs = torch.sigmoid(out)
            seq_preds.extend(probs.cpu().numpy().flatten())

    # Graph predictions on test
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    graph_preds = []
    with torch.no_grad():
        chunk_size = 10000
        for i in range(0, g_edge_label_index.size(1), chunk_size):
            chunk = g_edge_label_index[:, i:i + chunk_size]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            probs = torch.sigmoid(out)
            graph_preds.extend(probs.cpu().numpy().flatten())

    test_labels = np.array(final_labels)
    seq_preds_np = np.array(seq_preds)
    graph_preds_np = np.array(graph_preds)

    # Ensemble predictions with test bio features
    ensemble_preds = ensemble.predict(seq_preds_np, graph_preds_np, bio_features=test_bio_features, method="stacking")

    # Report metrics using the project's metrics reporter
    report_all_metrics(
        model_names=["ESM-MLP", "GAT", "Ensemble"],
        y_trues=[test_labels, test_labels, test_labels],
        y_probs_list=[seq_preds_np, graph_preds_np, ensemble_preds]
    )

    print("\n  Evaluation complete!")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all TransGraph-PPI models")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete old checkpoints for a clean start")
    parser.add_argument("--seq-epochs", type=int, default=50,
                        help="Epochs for sequence model (default: 50)")
    parser.add_argument("--gat-epochs", type=int, default=100,
                        help="Epochs for graph model (default: 100)")
    parser.add_argument("--skip-seq", action="store_true",
                        help="Skip sequence model training")
    parser.add_argument("--skip-gat", action="store_true",
                        help="Skip graph model training")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="Skip ensemble training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation (no training)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            TransGraph-PPI — Complete Training Pipeline              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Verify data exists
    required_files = [
        PROCESSED_DATA_DIR / "train.csv",
        PROCESSED_DATA_DIR / "val.csv",
        PROCESSED_DATA_DIR / "test.csv",
        PROCESSED_DATA_DIR / "embeddings.pt",
        PROCESSED_DATA_DIR / "ppi_graph.pt",
        PROCESSED_DATA_DIR / "ppi_graph_mapping.pt",
    ]
    for f in required_files:
        if not f.exists():
            print(f"ERROR: Required file missing: {f}")
            sys.exit(1)

    print("  All required data files found. ✓\n")

    if args.eval_only:
        evaluate_all()
        sys.exit(0)

    if args.fresh:
        print("  --fresh flag: clearing old checkpoints...")
        clear_checkpoints()

    # Phase 1: Sequence Model
    if not args.skip_seq:
        train_sequence_model(epochs=args.seq_epochs)
    else:
        print("\n  Skipping sequence model training (--skip-seq)")

    # Phase 2: Graph Model
    if not args.skip_gat:
        train_graph_model(epochs=args.gat_epochs)
    else:
        print("\n  Skipping graph model training (--skip-gat)")

    # Phase 3: Ensemble
    if not args.skip_ensemble:
        train_ensemble_model()
    else:
        print("\n  Skipping ensemble training (--skip-ensemble)")

    # Phase 4: Evaluate
    evaluate_all()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                   Training Pipeline Complete!                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
