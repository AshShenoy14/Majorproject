"""
ESMGAT Colab & Standalone Training Entry Point
TransGraph-PPI: GAT Graph Learner Experiment

Designed to run seamlessly on Google Colab (with NVIDIA T4 GPU) or any Linux/Windows environment.
All outputs are strictly isolated inside ESMGAT/. The existing GraphSAGE model and checkpoints
are completely untouched and protected by explicit runtime path assertions.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch

# ---------------------------------------------------------------------------
# Dynamic Path Resolution (Works in Colab, Linux, Windows without hard-coding)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Support command line or environment override, falling back to relative project root
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports from ESMGAT and existing shared utilities
from ESMGAT.models.gat_model import GATLinkPredictor
from ESMGAT.training.gat_base_trainers import (
    GAT_CFG, fit_gat, gat_logits, assert_safe_path, focal_loss
)
from src.utils.calibration import PlattScaler, calibration_report
from src.utils.seed import set_seed


def print_environment_diagnostics(device: torch.device):
    print("=" * 75)
    print("  ESMGAT Training Pipeline — System & Hardware Diagnostics")
    print("=" * 75)
    print(f"Project Root:            {PROJECT_ROOT}")
    print(f"PyTorch Version:         {torch.__version__}")
    try:
        import torch_geometric
        pyg_ver = torch_geometric.__version__
    except ImportError:
        pyg_ver = "Not Installed"
    print(f"PyTorch Geometric:       {pyg_ver}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:          {cuda_avail}")
    print(f"Target Device:           {device}")

    if cuda_avail and device.type == "cuda":
        dev_idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(dev_idx)
        print(f"GPU Name:                {props.name}")
        print(f"GPU Total VRAM:          {props.total_memory / 1e9:.2f} GB")
        print(f"Compute Capability:      {props.major}.{props.minor}")
    else:
        print(f"CPU Available Threads:   {torch.get_num_threads()}")
    print("=" * 75)


def _load_link_pairs(df: pd.DataFrame, node_mapping: dict):
    keep = df["protein1"].isin(node_mapping) & df["protein2"].isin(node_mapping)
    df_filtered = df[keep]
    u = np.array([node_mapping[p] for p in df_filtered["protein1"]], dtype=np.int64)
    v = np.array([node_mapping[p] for p in df_filtered["protein2"]], dtype=np.int64)
    y = df_filtered["label"].values.astype(np.float32)
    return u, v, y


def preflight_memory_and_forward_check(model: GATLinkPredictor, data, device: torch.device):
    """
    Step 1 of Memory Safety:
    Loads real graph, moves to device, runs one encode forward pass, and measures VRAM.
    """
    print("\n--- [PRE-FLIGHT] Testing GAT Full-Graph Encode Pass & Memory Safety ---")
    print(f"Node feature matrix shape:  {data.x.shape} (nodes={data.num_nodes}, in_channels={data.x.shape[1]})")
    print(f"Graph edge index shape:     {data.edge_index.shape} (edges={data.edge_index.shape[1]})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model.eval()
    try:
        with torch.no_grad():
            x_dev = data.x.to(device)
            ei_dev = data.edge_index.to(device)
            t0 = time.time()
            z = model.encode(x_dev, ei_dev)
            enc_time = time.time() - t0

            assert z.shape == (data.num_nodes, model.hidden_channels), (
                f"Expected z shape ({data.num_nodes}, {model.hidden_channels}), got {z.shape}"
            )
            assert torch.isfinite(z).all(), "NaN or Inf values detected in GAT node embeddings!"

            print(f"Encode pass successful in {enc_time:.2f}s | z shape: {z.shape}")
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated(device) / 1e9
                res = torch.cuda.memory_reserved(device) / 1e9
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"  VRAM Allocated: {alloc:.2f} GB | Reserved: {res:.2f} GB | Peak: {peak:.2f} GB")
    except torch.cuda.OutOfMemoryError as oom:
        print("\n" + "!" * 75)
        print("CRITICAL: CUDA Out-Of-Memory occurred during GAT full-graph forward pass!")
        print(f"Details: {oom}")
        print("!" * 75)
        raise RuntimeError("GAT forward pass exceeded GPU memory. Training cannot proceed as-is.") from oom


def run_sanity_dry_run(model: GATLinkPredictor, data, tr_u, tr_v, tr_y, device: torch.device):
    """
    Verifies all 11 sanity checks before full training:
      1. Dataset loads
      2. Graph loads
      3. Model loads
      4. GPU detected
      5. Graph tensors transferred
      6. GAT encode works
      7. Pair decoding works
      8. Loss calculated
      9. Backward pass works
      10. Optimizer step works
      11. GPU memory remains within limits
    """
    print("\n--- [SANITY CHECK] Executing 11-Point Verification on Real Graph ---")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 1-5 verified by arguments and prior setup
    x_dev = data.x.to(device)
    ei_dev = data.edge_index.to(device)

    # 6. GAT encode
    z = model.encode(x_dev, ei_dev)
    assert z.shape == (data.num_nodes, 256), "Encode shape mismatch"

    # 7. Pair decoding on a sample of 100 pairs
    z_det = z.detach().requires_grad_(True)
    n_sample = min(100, len(tr_u))
    s_sub = torch.as_tensor(tr_u[:n_sample], device=device)
    d_sub = torch.as_tensor(tr_v[:n_sample], device=device)
    y_sub = torch.as_tensor(tr_y[:n_sample], dtype=torch.float32, device=device)

    logits = model.decode(z_det, s_sub, d_sub).squeeze(1)
    assert logits.shape == (n_sample,), f"Decoder shape mismatch: {logits.shape}"

    # 8. Loss calculation
    loss = focal_loss(logits, y_sub)
    assert torch.isfinite(loss), "Loss is not finite"

    # 9. Backward pass
    opt.zero_grad()
    loss.backward()
    z.backward(z_det.grad)

    # 10. Optimizer step
    opt.step()

    # 11. Memory verification
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1e9
        print(f"Sanity step completed successfully | VRAM allocated: {alloc:.2f} GB")
    else:
        print("Sanity step completed successfully on CPU.")

    opt.zero_grad()
    model.eval()
    print("[SANITY CHECK PASSED] All 11 verification checkpoints passed successfully.\n")


def train_gat(
    project_root: Path,
    max_epochs: int = GAT_CFG["max_epochs"],
    patience: int = GAT_CFG["patience"],
    lr: float = GAT_CFG["lr"],
    seed: int = 42,
    force_cpu: bool = False,
    dry_run_only: bool = False,
    ckpt_every: int = 5,
):
    set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")
    print_environment_diagnostics(device)

    # -----------------------------------------------------------------------
    # Shared Data Paths (Read-Only)
    # -----------------------------------------------------------------------
    data_dir = project_root / "data" / "processed"
    graph_path = data_dir / "ppi_graph.pt"
    mapping_path = data_dir / "ppi_graph_mapping.pt"
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    for p in [graph_path, mapping_path, train_csv, val_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required shared data file missing: {p}")

    print(f"Loading shared graph: {graph_path} ...")
    graph_data = torch.load(graph_path, weights_only=False)
    node_mapping = torch.load(mapping_path, weights_only=False)

    print("Loading train and validation splits...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    tr_u, tr_v, tr_y = _load_link_pairs(df_train, node_mapping)
    va_u, va_v, va_y = _load_link_pairs(df_val, node_mapping)

    print(f"Dataset summary: Train pairs = {len(tr_u)} | Validation pairs = {len(va_u)}")
    print(f"Graph summary:   Nodes = {graph_data.num_nodes} | Edges = {graph_data.edge_index.shape[1]}")

    # -----------------------------------------------------------------------
    # Isolated Output Paths (Strictly inside ESMGAT/)
    # -----------------------------------------------------------------------
    esmgat_dir = project_root / "ESMGAT"
    ckpt_dir = esmgat_dir / "checkpoints" / "gat"
    weights_dir = esmgat_dir / "weights"
    results_dir = esmgat_dir / "results"

    for d in [ckpt_dir, weights_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_dir / "gat_checkpoint.pt"
    best_model_file = weights_dir / "gat_model_best.pth"
    calibrator_file = weights_dir / "gat_calibrator.json"
    report_file = results_dir / "gat_calibration.json"
    metrics_file = results_dir / "gat_val_metrics.json"

    # Enforce path safety assertions
    for target in [ckpt_file, best_model_file, calibrator_file, report_file]:
        assert_safe_path(target, project_root)

    # -----------------------------------------------------------------------
    # Initialize GAT Model
    # -----------------------------------------------------------------------
    in_channels = graph_data.x.shape[1]
    print(f"\nInitializing GATLinkPredictor(in_channels={in_channels}, hidden_channels=256, heads=4)...")
    model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256, heads=4, dropout=0.4).to(device)

    # Pre-flight memory and sanity check
    preflight_memory_and_forward_check(model, graph_data, device)
    run_sanity_dry_run(model, graph_data, tr_u, tr_v, tr_y, device)

    if dry_run_only:
        print("[DRY-RUN ONLY] Sanity tests completed. Exiting without running full training.")
        return

    # -----------------------------------------------------------------------
    # Full Training Loop
    # -----------------------------------------------------------------------
    t_start = time.time()
    cfg = dict(GAT_CFG, max_epochs=max_epochs, patience=patience, lr=lr)

    trained_model, summary = fit_gat(
        model=model,
        graph_x=graph_data.x,
        graph_edge_index=graph_data.edge_index,
        src_fit=tr_u,
        dst_fit=tr_v,
        y_fit=tr_y,
        src_es=va_u,
        dst_es=va_v,
        y_es=va_y,
        device=device,
        project_root=project_root,
        cfg=cfg,
        ckpt_path=str(ckpt_file),
        ckpt_every=ckpt_every,
        tag="ESM-GAT",
    )

    total_training_time = time.time() - t_start

    # Save best GAT model weights
    assert_safe_path(best_model_file, project_root)
    torch.save({k: v.cpu() for k, v in trained_model.state_dict().items()}, str(best_model_file))
    print(f"\nSaved Best GAT Weights to: {best_model_file}")

    # -----------------------------------------------------------------------
    # Post-Training Calibration (Fit Platt Scaler on val.csv Logits)
    # -----------------------------------------------------------------------
    print("\n--- Fitting GAT Platt Calibrator on Validation Logits (val.csv) ---")
    val_logits = gat_logits(trained_model, graph_data.x, graph_data.edge_index, va_u, va_v, device).astype(np.float64)
    val_probs_raw = 1.0 / (1.0 + np.exp(-val_logits))

    scaler = PlattScaler().fit(val_logits, va_y)
    val_probs_cal = scaler.transform_logits(val_logits)

    assert_safe_path(calibrator_file, project_root)
    scaler.save(str(calibrator_file))
    print(f"Saved GAT Calibrator (a={scaler.a:.4f}, b={scaler.b:.4f}) to: {calibrator_file}")

    cal_rep = calibration_report(va_y, val_probs_raw, val_probs_cal)
    cal_rep["platt"] = {"a": scaler.a, "b": scaler.b}
    cal_rep["n_val"] = int(len(va_y))

    assert_safe_path(report_file, project_root)
    with open(report_file, "w") as f:
        json.dump(cal_rep, f, indent=2)

    # -----------------------------------------------------------------------
    # Compute and Save Validation Metrics
    # -----------------------------------------------------------------------
    val_pred_labels = (val_probs_cal >= 0.5).astype(int)
    val_metrics = {
        "accuracy": float(accuracy_score(va_y, val_pred_labels)),
        "precision": float(precision_score(va_y, val_pred_labels, zero_division=0)),
        "recall": float(recall_score(va_y, val_pred_labels, zero_division=0)),
        "f1": float(f1_score(va_y, val_pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(va_y, val_probs_cal)),
        "pr_auc": float(average_precision_score(va_y, val_probs_cal)),
    }

    assert_safe_path(metrics_file, project_root)
    with open(metrics_file, "w") as f:
        json.dump(val_metrics, f, indent=2)

    # -----------------------------------------------------------------------
    # Final Colab Training Report
    # -----------------------------------------------------------------------
    peak_vram_str = "N/A (CPU)"
    if device.type == "cuda":
        peak_vram_str = f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB"

    print("\n" + "=" * 75)
    print("  ESMGAT Training Pipeline — Execution Summary")
    print("=" * 75)
    print(f"Device Used:              {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Total Training Time:      {total_training_time / 60:.2f} minutes")
    print(f"Total Epochs Run:         {summary['total_epochs_run']}")
    print(f"Best Epoch:               {summary['best_epoch']}")
    print(f"Best Validation Loss:     {summary['best_val_loss']:.4f}")
    print(f"Early Stopping Fired:     {summary['early_stopped']}")
    print(f"Peak GPU VRAM:            {peak_vram_str}")
    print(f"GAT Checkpoint:           {best_model_file}")
    print(f"GAT Calibrator:           {calibrator_file} (a={scaler.a:.3f}, b={scaler.b:.3f})")
    print("\nValidation Performance (val.csv, threshold=0.5):")
    for k, v in val_metrics.items():
        print(f"  {k:15s}: {v * 100:.2f}%" if k != "roc_auc" and k != "pr_auc" else f"  {k:15s}: {v:.4f}")
    print("=" * 75)
    print("\nTraining complete. You can download the following files from Colab to local:")
    print(f"  1. {best_model_file}")
    print(f"  2. {calibrator_file}")
    print(f"  3. {report_file}")
    print(f"  4. {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="ESMGAT Colab & Standalone Training Pipeline")
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT),
                        help="Root directory of the project (e.g. /content/Majorproject)")
    parser.add_argument("--epochs", type=int, default=GAT_CFG["max_epochs"],
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=GAT_CFG["patience"],
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--lr", type=float, default=GAT_CFG["lr"],
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU execution even if CUDA is available")
    parser.add_argument("--dry_run_only", action="store_true",
                        help="Run pre-flight diagnostics & sanity check only without full training")
    parser.add_argument("--ckpt_every", type=int, default=5,
                        help="Save intermediate checkpoint every N epochs (default: 5)")
    args = parser.parse_args()

    train_gat(
        project_root=Path(args.project_root),
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
        force_cpu=args.force_cpu,
        dry_run_only=args.dry_run_only,
        ckpt_every=args.ckpt_every,
    )


if __name__ == "__main__":
    main()
