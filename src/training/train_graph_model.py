import os
import sys
# Set project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Importing paths first redirects HF/torch caches to ./.cache before any model download
from src.utils.paths import PROCESSED_DATA_DIR, CHECKPOINT_DIR, MODELS_DIR
from src.utils.seed import set_seed
from src.models.graph_model import SAGELinkPredictor, GINLinkPredictor
from src.training.base_trainers import GRAPH_CFG, get_device, fit_graph, graph_logits
from src.utils.calibration import PlattScaler, calibration_report, cross_fitted_probs, logit
import json
import torch
import argparse
import time
import pandas as pd
import numpy as np


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


def _pairs(df, node_mapping):
    keep = df["protein1"].isin(node_mapping) & df["protein2"].isin(node_mapping)
    df = df[keep]
    return (np.array([node_mapping[p] for p in df["protein1"]]),
            np.array([node_mapping[p] for p in df["protein2"]]),
            df["label"].values)


def train(
    epochs: int = GRAPH_CFG["max_epochs"],
    lr: float = GRAPH_CFG["lr"],
    graph_path: str = None,
    force_cpu: bool = False,
    cpu_threads: int = None,
    model_type: str = "SAGE",
    checkpoint_dir: str = str(CHECKPOINT_DIR),
    model_dir: str = str(MODELS_DIR),
    ckpt_every: int = 5,
    seed: int = 42,
    report_dir: str = None,
):
    """
    Train the GNN link predictor with full-batch training, using the same routine as the OOF fold models
    (base_trainers.fit_graph): focal loss, AdamW, cosine annealing, gradient clipping, early stopping on
    validation BCE. Early stopping uses val.csv; test.csv is never opened here.
    model_type: 'SAGE' (GraphSAGE / SAGEConv, no attention) or 'GIN'.

    Training checkpoint (resumable):  <checkpoint_dir>/graph_checkpoint.pt  (every `ckpt_every` epochs)
    Best model:                       <model_dir>/graph_model_best.pth
    """
    set_seed(seed)
    device = configure_runtime(force_cpu=force_cpu, cpu_threads=cpu_threads)
    t_start = time.time()
    print(f"Training {model_type} link predictor on {device} (full-batch)...")

    if not (graph_path and os.path.exists(graph_path)):
        print("Graph file not found.")
        return
    data = torch.load(graph_path, weights_only=False)
    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if not os.path.exists(mapping_path):
        print("Node mapping not found.")
        return
    node_mapping = torch.load(mapping_path, weights_only=False)

    tr_u, tr_v, tr_y = _pairs(pd.read_csv(PROCESSED_DATA_DIR / "train.csv"), node_mapping)
    va_u, va_v, va_y = _pairs(pd.read_csv(PROCESSED_DATA_DIR / "val.csv"), node_mapping)
    print(f"Link-prediction pairs: {len(tr_u)} train / {len(va_u)} val | nodes {data.num_nodes} | max epochs {epochs}")

    if model_type == "GIN":
        model = GINLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128).to(device)
    else:
        model = SAGELinkPredictor(in_channels=data.x.shape[1], hidden_channels=256).to(device)

    cfg = dict(GRAPH_CFG, max_epochs=epochs, lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model = fit_graph(
        model, data.x, data.edge_index, tr_u, tr_v, tr_y, va_u, va_v, va_y, device, cfg=cfg,
        ckpt_path=os.path.join(checkpoint_dir, "graph_checkpoint.pt"), ckpt_every=ckpt_every, tag="graph",
    )
    best_path = os.path.join(model_dir, "graph_model_best.pth")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)
    print(f"\nGraph model training complete in {(time.time() - t_start) / 60:.1f} min. Saved {best_path}")

    # ---- Calibration: Platt scaling on VALIDATION logits only (test.csv untouched) ----
    val_logits = graph_logits(model, data.x, data.edge_index, va_u, va_v, device).astype(np.float64)
    p_raw = 1.0 / (1.0 + np.exp(-val_logits))
    scaler = PlattScaler().fit(val_logits, va_y)
    p_cal = scaler.transform_logits(val_logits)
    p_cf = cross_fitted_probs(val_logits, va_y, seed=seed)  # out-of-sample estimate (scaler has only 2 parameters)
    report = calibration_report(va_y, p_raw, p_cal)
    report["after_cross_fitted_on_val"] = calibration_report(va_y, p_raw, p_cf)["after"]
    report["platt"] = {"a": scaler.a, "b": scaler.b}
    report["n_val"] = int(len(va_y))
    report["note"] = "Calibrator fit and scored on val.csv; the cross-fitted row is the out-of-sample estimate."
    scaler.save(os.path.join(model_dir, "graph_calibrator.json"))
    report_dir = report_dir or str(MODELS_DIR.parent / "assets" / "evaluation")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "graph_calibration.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("GraphSAGE calibration on val.csv:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=GRAPH_CFG["max_epochs"], help="Maximum epochs (early stopping applies).")
    parser.add_argument("--lr", type=float, default=GRAPH_CFG["lr"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Maximum PyTorch CPU threads (default: half logical cores)")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR),
                        help="Directory for resumable training checkpoints (mount Google Drive here on Colab)")
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR),
                        help="Directory the best model is written to")
    parser.add_argument("--ckpt_every", type=int, default=5, help="Save a training checkpoint every N epochs")
    parser.add_argument("--report_dir", type=str, default=None, help="Where graph_calibration.json is written")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="SAGE", choices=["SAGE", "GIN", "GAT"],
                        help="SAGE = GraphSAGE (SAGEConv, no attention). 'GAT' is a deprecated alias for SAGE.")
    args = parser.parse_args()

    if args.model_type == "GAT":
        print("Warning: --model_type GAT is a deprecated alias; the model is GraphSAGE (no attention). Using SAGE.")
        args.model_type = "SAGE"

    train(
        epochs=args.epochs,
        lr=args.lr,
        graph_path=args.graph_path,
        force_cpu=args.force_cpu,
        cpu_threads=args.cpu_threads,
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
        report_dir=args.report_dir,
    )
