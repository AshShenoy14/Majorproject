import torch
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.seed import set_seed
from src.models.sequence_model import SequencePPIModel
from src.training.base_trainers import SEQ_CFG, get_device, build_embedding_table, fit_sequence
from src.utils.paths import PROCESSED_DATA_DIR, CHECKPOINT_DIR, MODELS_DIR


def configure_runtime(force_cpu: bool = False, cpu_threads: int = None):
    """CUDA when available (Colab T4), otherwise CPU; also sets CPU thread counts for thermally stable CPU runs."""
    device = get_device(force_cpu)
    if device.type == "cpu":
        if cpu_threads is None:
            cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        cpu_threads = max(1, int(cpu_threads))
        try:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
        except RuntimeError:
            pass  # parallel work already started, cannot change threads
        print(f"Runtime: CPU | torch threads={torch.get_num_threads()}")
    else:
        print(f"Runtime: CUDA ({torch.cuda.get_device_name(0)})")
    return device


def _load_split(name, embeddings):
    df = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
    keep = df["protein1"].isin(embeddings) & df["protein2"].isin(embeddings)
    if not keep.all():
        print(f"Warning: dropped {(~keep).sum()} {name} pairs with missing embeddings.")
    return df[keep].reset_index(drop=True)


def train(
    epochs: int = SEQ_CFG["max_epochs"],
    batch_size: int = SEQ_CFG["batch_size"],
    lr: float = SEQ_CFG["lr"],
    embedding_path: str = None,
    force_cpu: bool = False,
    cpu_threads: int = None,
    checkpoint_dir: str = str(CHECKPOINT_DIR),
    model_dir: str = str(MODELS_DIR),
    ckpt_every: int = 5,
    seed: int = 42,
):
    """
    Train the Sequence PPI Model. Uses the same routine as the OOF fold models (base_trainers.fit_sequence):
    focal loss, AdamW, cosine annealing, gradient clipping, early stopping on validation focal loss.
    Early stopping uses val.csv; test.csv is never opened here.

    Training checkpoint (resumable):  <checkpoint_dir>/sequence_checkpoint.pt  (every `ckpt_every` epochs)
    Best model:                       <model_dir>/sequence_model_best.pth
    """
    set_seed(seed)
    device = configure_runtime(force_cpu=force_cpu, cpu_threads=cpu_threads)
    t_start = time.time()

    if not (embedding_path and os.path.exists(embedding_path)):
        print("No embedding file provided/found. Cannot proceed without embeddings.")
        return
    embeddings = torch.load(embedding_path, weights_only=False)

    train_df = _load_split("train", embeddings)
    val_df = _load_split("val", embeddings)
    print(f"Dataset: {len(train_df)} train / {len(val_df)} val samples")

    proteins = sorted(set(train_df["protein1"]) | set(train_df["protein2"]) | set(val_df["protein1"]) | set(val_df["protein2"]))
    table, row_of = build_embedding_table(embeddings, proteins, device)
    idx = lambda df, col: np.array([row_of[p] for p in df[col]])
    print(f"Feature dimension: {table.shape[1]} | batch size {batch_size} | max epochs {epochs}")

    cfg = dict(SEQ_CFG, max_epochs=epochs, batch_size=batch_size, lr=lr)
    model = SequencePPIModel(input_dim=table.shape[1]).to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model = fit_sequence(
        model, table,
        idx(train_df, "protein1"), idx(train_df, "protein2"), train_df["label"].values,
        idx(val_df, "protein1"), idx(val_df, "protein2"), val_df["label"].values,
        device, cfg=cfg, ckpt_path=os.path.join(checkpoint_dir, "sequence_checkpoint.pt"),
        ckpt_every=ckpt_every, tag="seq",
    )
    best_path = os.path.join(model_dir, "sequence_model_best.pth")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)
    print(f"\nSequence model training complete in {(time.time() - t_start) / 60:.1f} min. Saved {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=SEQ_CFG["max_epochs"], help="Maximum epochs (early stopping applies).")
    parser.add_argument("--batch_size", type=int, default=SEQ_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=SEQ_CFG["lr"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Maximum PyTorch CPU threads (default: half logical cores)")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR),
                        help="Directory for resumable training checkpoints (mount Google Drive here on Colab)")
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR),
                        help="Directory the best model is written to")
    parser.add_argument("--ckpt_every", type=int, default=5, help="Save a training checkpoint every N epochs")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to dictionary of protein embeddings (.pt)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_path=args.embedding_path,
        force_cpu=args.force_cpu,
        cpu_threads=args.cpu_threads,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
    )
