import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.utils.dataset import PPIDataset
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR, MODELS_DIR


def train(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, embedding_path: str = None):
    """
    Train the Sequence PPI Model with checkpoint-based resume and best-model saving.

    Checkpoint saved to: checkpoints/sequence_checkpoint.pt  (overwritten each epoch)
    Best model saved to: models/sequence_model_best.pth      (lowest validation loss)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training Sequence Model on {device}...")

    # ── Load Embeddings ──────────────────────────────────────────────────
    if embedding_path and os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, weights_only=False)
        # Keep embeddings in their native dtype (float16) to save ~7.5 GB RAM
        # Conversion to float32 happens per-sample in the Dataset's __getitem__
    else:
        print("No embedding file provided/found. Cannot proceed without embeddings.")
        return

    # ── Datasets & DataLoaders ───────────────────────────────────────────
    train_dataset = PPIDataset(PROCESSED_DATA_DIR / "train.csv", embeddings)
    val_dataset   = PPIDataset(PROCESSED_DATA_DIR / "val.csv", embeddings)

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
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[0]

    model     = SequencePPIModel(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # ── Resume from checkpoint if it exists ──────────────────────────────
    checkpoint_path = CHECKPOINT_DIR / "sequence_checkpoint.pt"
    start_epoch     = 0
    best_val_loss   = float("inf")

    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch   = checkpoint["epoch"] + 1          # start from next epoch
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
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
            loss = criterion(outputs, labels)
            loss.backward()
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
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc      = val_correct / val_total if val_total > 0 else 0.0

        # --- Epoch summary ---
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )
        print(f"  Progress: {epoch+1}/{epochs} epochs done "
              f"({(epoch+1)/epochs*100:.0f}%) — {epochs - epoch - 1} remaining")

        # --- Save best model if validation loss improved ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved → {best_model_path}")

        # --- Save checkpoint (overwrite each epoch) ---
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "best_val_loss": best_val_loss,
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved → {checkpoint_path}")

    print(f"\nSequence Model training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to dictionary of protein embeddings (.pt)")
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr, args.embedding_path)
