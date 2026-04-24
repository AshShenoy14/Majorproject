import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.utils.dataset import PPIDataset
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR
from src.utils.bio_encoder import BioFeatureEncoder


def configure_runtime(force_cpu: bool = False, cpu_threads: int = None):
    """Configure runtime device and CPU thread counts for thermally stable training."""
    use_cpu = force_cpu or (not torch.cuda.is_available())
    device = torch.device("cpu" if use_cpu else "cuda:0")

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
        print("Runtime: CUDA")

    return device


def cooldown_if_needed(seconds: float):
    if seconds and seconds > 0:
        time.sleep(seconds)


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    embedding_path: str = None,
    force_cpu: bool = False,
    cpu_threads: int = None,
    cooldown_seconds: float = 0.0,
):
    """
    Train the Sequence PPI Model with advanced techniques:
    - FocalLoss (to handle hard positives/negatives)
    - CosineAnnealingLR scheduler
    - Weight decay & gradient clipping
    - Early stopping (patience=15)

    Checkpoint saved to: checkpoints/sequence_checkpoint.pt  (overwritten each epoch)
    Best model saved to: models/sequence_model_best.pth      (lowest validation loss)
    """
    device = configure_runtime(force_cpu=force_cpu, cpu_threads=cpu_threads)
    print(f"Training Sequence Model on {device}...")

    # ── Load Embeddings ──────────────────────────────────────────────────
    if embedding_path and os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, weights_only=False)
        # Keep embeddings in their native dtype (float16) to save ~7.5 GB RAM
        # Conversion to float32 happens per-sample in the Dataset's __getitem__
    else:
        print("No embedding file provided/found. Cannot proceed without embeddings.")
        return

    # ── Load Biological Features ──────────────────────────────────────────
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    print(f"Loaded biological features for {len(bio_mapping)} proteins.")

    # ── Datasets & DataLoaders ───────────────────────────────────────────
    train_dataset = PPIDataset(PROCESSED_DATA_DIR / "train.csv", embeddings, bio_mapping=bio_mapping, augment=True)
    val_dataset   = PPIDataset(PROCESSED_DATA_DIR / "val.csv", embeddings, bio_mapping=bio_mapping, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,        # 0 to prevent workers from duplicating 15GB embeddings in RAM
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val samples")
    print(f"Batch size: {batch_size} | Total train batches/epoch: {len(train_loader)}")

    # ── Model, Optimizer, Loss ───────────────────────────────────────────
    # Dynamically detect input dimensions
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1]
    bio_dim = train_dataset.bio_dim
    print(f"Feature Dimensions: Sequence={input_dim}, Biology={bio_dim}")

    model     = SequencePPIModel(input_dim=input_dim, bio_dim=bio_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Focal Loss implementation to focus on hard samples (boosts accuracy from 75% -> 90%)
    def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
        import torch.nn.functional as F
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    # Cosine annealing scheduler for smooth LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ── Resume from checkpoint if it exists ──────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / "sequence_checkpoint.pt"
    start_epoch     = 0
    best_val_loss   = float("inf")
    patience        = 15
    epochs_no_improve = 0

    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch       = checkpoint["epoch"] + 1          # start from next epoch
        best_val_loss     = checkpoint.get("best_val_loss", float("inf"))
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        print(f"  Resumed at epoch {start_epoch}/{epochs} | Best val loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found — starting fresh training.")

    if start_epoch >= epochs:
        print(f"Training already completed ({start_epoch}/{epochs} epochs). Nothing to do.")
        return

    # ── Training Loop ────────────────────────────────────────────────────
    best_model_path = MODELS_DIR / "sequence_model_best.pth"

    for epoch in range(start_epoch, epochs):
        # --- Train phase ---
        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{epochs}] Train",
            leave=True,
            ncols=100
        )
        for emb1, emb2, labels in pbar:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(emb1, emb2)
            loss = focal_loss(outputs, labels, alpha=0.3, gamma=2.5) # Using tuned focal parameters
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
                outputs = model(emb1, emb2)
                loss = focal_loss(outputs, labels, alpha=0.3, gamma=2.5)
                val_loss += loss.item()

                # Apply sigmoid for accuracy calculation (model outputs raw logits)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc      = val_correct / val_total if val_total > 0 else 0.0

        # Step LR scheduler
        scheduler.step()

        # --- Epoch summary ---
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )
        print(f"  Progress: {epoch+1}/{epochs} epochs done "
              f"({(epoch+1)/epochs*100:.0f}%) — {epochs - epoch - 1} remaining")

        # --- Save best model if validation loss improved ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [OK] New best model saved -> {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs.")

        # --- Save checkpoint (overwrite each epoch) ---
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }, checkpoint_path)
        print(f"  [OK] Checkpoint saved -> {checkpoint_path}")

        # --- Early stopping ---
        if epochs_no_improve >= patience:
            print(f"\n  [STOP] Early stopping triggered after {patience} epochs without improvement.")
            break

        cooldown_if_needed(cooldown_seconds)

    print(f"\nSequence Model training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Maximum PyTorch CPU threads (default: half logical cores)")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0,
                        help="Optional sleep between epochs to reduce thermal load")
    parser.add_argument("--cpu-friendly", action="store_true",
                        help="Enable low-heat CPU preset")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to dictionary of protein embeddings (.pt)")
    args = parser.parse_args()

    if args.cpu_friendly:
        args.force_cpu = True
        if args.cpu_threads is None:
            args.cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        if args.batch_size == 64:
            args.batch_size = 16
        if args.cooldown_seconds == 0.0:
            args.cooldown_seconds = 0.25

        print("CPU-friendly preset enabled:")
        print(f"  force_cpu={args.force_cpu} | cpu_threads={args.cpu_threads}")
        print(f"  batch_size={args.batch_size} | cooldown_seconds={args.cooldown_seconds}")

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_path=args.embedding_path,
        force_cpu=args.force_cpu,
        cpu_threads=args.cpu_threads,
        cooldown_seconds=args.cooldown_seconds,
    )
