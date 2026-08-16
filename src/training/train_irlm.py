import os
import sys
import json
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.irlm_module import InteractionRegionLocalizationModule, IRLMLoss


def seed_everything(seed: int = 42):
    """Sets deterministic seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IRLMDataset(torch.utils.data.Dataset):
    """
    Dataset loader for processed IRLM .npz structural complex artifacts.
    """
    def __init__(self, file_paths: List[str]):
        self.file_paths = sorted(file_paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.file_paths[idx]
        data = np.load(path)

        h_a = torch.tensor(data['esm_embedding_a'], dtype=torch.float32)
        h_b = torch.tensor(data['esm_embedding_b'], dtype=torch.float32)
        contact_map = torch.tensor(data['contact_map'], dtype=torch.float32)
        mask_a = torch.tensor(data['interface_mask_a'], dtype=torch.float32)
        mask_b = torch.tensor(data['interface_mask_b'], dtype=torch.float32)

        # Graph embeddings (optional)
        z_a = torch.tensor(data['graph_embedding_a'], dtype=torch.float32) if 'graph_embedding_a' in data else None
        z_b = torch.tensor(data['graph_embedding_b'], dtype=torch.float32) if 'graph_embedding_b' in data else None

        # Sanity Checks
        assert not torch.isnan(h_a).any() and not torch.isinf(h_a).any(), f"NaN/Inf in h_a ({path})"
        assert not torch.isnan(h_b).any() and not torch.isinf(h_b).any(), f"NaN/Inf in h_b ({path})"
        assert not torch.isnan(contact_map).any() and not torch.isinf(contact_map).any(), f"NaN/Inf in contact_map ({path})"

        # Verify binary contact map
        unique_vals = torch.unique(contact_map)
        assert torch.all((unique_vals == 0.0) | (unique_vals == 1.0)), f"Contact map must be binary (0 or 1) in {path}"

        # Verify dimension matching
        L_A, L_B = contact_map.shape
        assert h_a.shape[0] == L_A, f"h_a len {h_a.shape[0]} != contact_map L_A {L_A} ({path})"
        assert h_b.shape[0] == L_B, f"h_b len {h_b.shape[0]} != contact_map L_B {L_B} ({path})"
        assert mask_a.shape[0] == L_A, f"mask_a len {mask_a.shape[0]} != L_A {L_A} ({path})"
        assert mask_b.shape[0] == L_B, f"mask_b len {mask_b.shape[0]} != L_B {L_B} ({path})"

        return {
            "name": Path(path).stem,
            "h_a": h_a,
            "h_b": h_b,
            "contact_map": contact_map,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "z_a": z_a,
            "z_b": z_b
        }


def compute_contact_metrics_sweep(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    threshold_min: float = 0.001,
    threshold_max: float = 0.20,
    num_thresholds: int = 200
) -> Dict[str, float]:
    """
    Computes AUPRC and sweeps thresholds from threshold_min to threshold_max (200 thresholds)
    to find the threshold that maximizes validation F1 score on global predictions.
    """
    if len(all_probs) == 0 or len(all_targets) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auprc": 0.0,
            "best_threshold": float(threshold_min)
        }

    # Global AUPRC computation
    try:
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(all_targets, all_probs) if np.sum(all_targets) > 0 else 0.0
    except Exception:
        auprc = 0.0

    thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thresh = float(threshold_min)

    all_targets_bool = (all_targets == 1.0)

    for t in thresholds:
        pred_bin = (all_probs >= t)
        tp = np.sum(pred_bin & all_targets_bool)
        fp = np.sum(pred_bin & (~all_targets_bool))
        fn = np.sum((~pred_bin) & all_targets_bool)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_prec = precision
            best_rec = recall
            best_thresh = float(t)

    return {
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "auprc": float(auprc),
        "best_threshold": float(best_thresh)
    }


def train_one_epoch(model: nn.Module, 
                    dataset: IRLMDataset, 
                    optimizer: optim.Optimizer, 
                    loss_fn: IRLMLoss, 
                    device: torch.device) -> Dict[str, float]:
    """
    Executes one training epoch with batch_size=1 for variable sequence length safety.
    """
    model.train()
    total_loss = 0.0
    all_probs_list = []
    all_targets_list = []

    for i in range(len(dataset)):
        sample = dataset[i]
        h_a = sample["h_a"].to(device)
        h_b = sample["h_b"].to(device)
        cmap = sample["contact_map"].to(device)
        z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
        z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None

        optimizer.zero_grad()

        # Forward pass through IRLM
        r_a, r_b, interaction_matrix = model.compute_residue_importance(h_a, h_b, z_a, z_b)

        # Compute Loss
        loss = loss_fn(
            pred_prob=None,
            target_label=None,
            interaction_matrix=interaction_matrix,
            r_a=r_a,
            r_b=r_b,
            contact_map=cmap
        )

        assert not torch.isnan(loss) and not torch.isinf(loss), f"Training loss is NaN or Inf for sample {sample['name']}"

        # Backward pass
        loss.backward()

        # Check gradient validity
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {param_name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {param_name}"

        optimizer.step()

        total_loss += loss.item()
        all_probs_list.append(interaction_matrix.detach().cpu().numpy().flatten())
        all_targets_list.append(cmap.detach().cpu().numpy().flatten())

    avg_loss = total_loss / max(1, len(dataset))
    all_probs = np.concatenate(all_probs_list) if all_probs_list else np.array([])
    all_targets = np.concatenate(all_targets_list) if all_targets_list else np.array([])

    metrics = compute_contact_metrics_sweep(all_probs, all_targets)
    metrics["loss"] = float(avg_loss)
    return metrics


def evaluate(model: nn.Module, 
             dataset: IRLMDataset, 
             loss_fn: IRLMLoss, 
             device: torch.device) -> Dict[str, float]:
    """
    Evaluates IRLM model on validation dataset using a threshold sweep (0.001 to 0.20 across 200 values).
    """
    model.eval()
    total_loss = 0.0
    all_probs_list = []
    all_targets_list = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            h_a = sample["h_a"].to(device)
            h_b = sample["h_b"].to(device)
            cmap = sample["contact_map"].to(device)
            z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
            z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None

            r_a, r_b, interaction_matrix = model.compute_residue_importance(h_a, h_b, z_a, z_b)

            loss = loss_fn(
                pred_prob=None,
                target_label=None,
                interaction_matrix=interaction_matrix,
                r_a=r_a,
                r_b=r_b,
                contact_map=cmap
            )

            total_loss += loss.item()
            all_probs_list.append(interaction_matrix.detach().cpu().numpy().flatten())
            all_targets_list.append(cmap.detach().cpu().numpy().flatten())

    avg_loss = total_loss / max(1, len(dataset))
    all_probs = np.concatenate(all_probs_list) if all_probs_list else np.array([])
    all_targets = np.concatenate(all_targets_list) if all_targets_list else np.array([])

    metrics = compute_contact_metrics_sweep(all_probs, all_targets)
    metrics["loss"] = float(avg_loss)
    return metrics


def run_dry_run(data_files: List[str], device: torch.device, args: argparse.Namespace):
    """
    Executes a 2-example dry-run: loads 2 samples, performs forward, loss, backward, and optimizer step, then exits.
    """
    print("\n" + "=" * 60)
    print("      EXECUTING IRLM TRAINING PIPELINE DRY-RUN MODE")
    print("=" * 60)

    # Use 2 examples (duplicate if only 1 file is present)
    if len(data_files) == 1:
        dry_files = [data_files[0], data_files[0]]
        print(f"[DRY-RUN] Found 1 dataset file ({Path(data_files[0]).name}). Duplicating for 2-example dry run.")
    else:
        dry_files = data_files[:2]
        print(f"[DRY-RUN] Selected 2 dataset files: {[Path(f).name for f in dry_files]}")

    dataset = IRLMDataset(dry_files)
    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = IRLMLoss(
        lambda_contact=args.lambda_contact,
        lambda_sparsity=args.lambda_sparsity,
        lambda_smooth=args.lambda_smooth
    ).to(device)

    # Track parameter norm before step
    param_before = sum(p.norm().item() for p in model.parameters())

    print("\n--- Dry Run Iterations ---")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        h_a = sample["h_a"].to(device)
        h_b = sample["h_b"].to(device)
        cmap = sample["contact_map"].to(device)
        z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
        z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None

        optimizer.zero_grad()

        # Forward
        r_a, r_b, matrix = model.compute_residue_importance(h_a, h_b, z_a, z_b)

        # Loss
        loss = loss_fn(
            pred_prob=None,
            target_label=None,
            interaction_matrix=matrix,
            r_a=r_a,
            r_b=r_b,
            contact_map=cmap
        )

        print(f"Sample {idx+1} ({sample['name']}):")
        print(f"  - Protein A shape: {list(h_a.shape)}")
        print(f"  - Protein B shape: {list(h_b.shape)}")
        print(f"  - Contact Map shape: {list(cmap.shape)} (positive contacts: {int(torch.sum(cmap).item())})")
        print(f"  - Loss: {loss.item():.4f}")

        assert not torch.isnan(loss) and not torch.isinf(loss), "Dry-run loss is NaN/Inf!"

        # Backward
        loss.backward()

        # Check gradients
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  - Gradient Norm: {grad_norm:.4f}")
        assert grad_norm > 0, "Dry-run gradient norm is 0!"

        # Optimizer step
        optimizer.step()

    param_after = sum(p.norm().item() for p in model.parameters())
    param_diff = abs(param_after - param_before)

    print("\n--- Dry-Run Summary ---")
    print(f"Initial Parameter Norm : {param_before:.6f}")
    print(f"Final Parameter Norm   : {param_after:.6f}")
    print(f"Norm Difference        : {param_diff:.6f}")
    print("Status                 : SUCCESSful (Gradients update model parameters)")
    print("Checkpoint Saved       : NONE (Dry run mode - no checkpoint generated)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Interaction Region Localization Module (IRLM)")
    parser.add_argument("--data_dir", type=str, default="data/processed/irlm_dataset", help="Directory containing .npz complex artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation dataset split ratio")
    parser.add_argument("--lambda_contact", type=float, default=1.0, help="Weight for contact-map loss")
    parser.add_argument("--lambda_sparsity", type=float, default=0.05, help="Weight for 2D sparsity loss")
    parser.add_argument("--lambda_smooth", type=float, default=0.05, help="Weight for 1D smoothness loss")
    parser.add_argument("--dry_run", action="store_true", help="Execute 2-example dry run without full training or checkpoint saving")

    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover dataset files
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not data_files:
        raise FileNotFoundError(f"No .npz artifacts found in dataset directory: {args.data_dir}")

    print(f"Discovered {len(data_files)} dataset artifacts in {args.data_dir}")

    # Dry-Run Mode Execution
    if args.dry_run:
        run_dry_run(data_files, device, args)
        return

    # Train / Validation Split
    if len(data_files) < 2:
        raise ValueError(f"Insufficient dataset size ({len(data_files)}) for train/validation splitting. At least 2 complexes required.")

    rng = random.Random(args.seed)
    shuffled_files = list(data_files)
    rng.shuffle(shuffled_files)

    val_size = max(1, int(len(shuffled_files) * args.val_ratio))
    val_files = shuffled_files[:val_size]
    train_files = shuffled_files[val_size:]

    print(f"Dataset split: {len(train_files)} training, {len(val_files)} validation")

    train_dataset = IRLMDataset(train_files)
    val_dataset = IRLMDataset(val_files)

    # Model, Optimizer, Loss Setup
    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = IRLMLoss(
        lambda_contact=args.lambda_contact,
        lambda_sparsity=args.lambda_sparsity,
        lambda_smooth=args.lambda_smooth
    ).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("\nStarting IRLM Supervised Training...")
    for epoch in range(1, args.epochs + 1):
        train_res = train_one_epoch(model, train_dataset, optimizer, loss_fn, device)
        val_res = evaluate(model, val_dataset, loss_fn, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_precision": train_res["precision"],
            "train_recall": train_res["recall"],
            "train_f1": train_res["f1"],
            "train_auprc": train_res["auprc"],
            "train_best_threshold": train_res["best_threshold"],
            "val_loss": val_res["loss"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_f1": val_res["f1"],
            "val_auprc": val_res["auprc"],
            "val_best_threshold": val_res["best_threshold"],
        }
        history.append(epoch_record)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_res['loss']:.4f} (F1: {train_res['f1']:.4f}) | "
              f"Val Loss: {val_res['loss']:.4f} (F1: {val_res['f1']:.4f}, Best Thresh: {val_res['best_threshold']:.4f}, AUPRC: {val_res['auprc']:.4f})")

        # Save Best Model
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            torch.save(model.state_dict(), "models/irlm_best.pth")
            print(f"  --> Saved new best model to models/irlm_best.pth (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={args.patience}).")
                break

    # Save training history
    history_path = "data/processed/irlm_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")


if __name__ == "__main__":
    main()
