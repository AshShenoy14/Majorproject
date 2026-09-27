"""
Shared training routines and hyper-parameters for the GAT link predictor.

Maintains identical loss, optimizer, scheduler, early stopping, and two-stage
detached-gradient mechanics as the existing GraphSAGE trainer (src/training/base_trainers.py)
to ensure a fair, controlled scientific comparison.
"""
import copy
import os
from pathlib import Path
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAT_CFG = dict(
    max_epochs=100,
    patience=15,
    lr=1e-3,
    weight_decay=1e-4,
    eta_min=1e-6,
    focal_alpha=0.3,
    focal_gamma=2.5,
    chunk_size=1000,
    loss_scale=10.0,
    pair_weight=0.5,
)
EVAL_CHUNK = 2000


def get_device(force_cpu: bool = False) -> torch.device:
    """CUDA when available (e.g. Colab T4), otherwise CPU."""
    return torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, gamma: float = 2.5) -> torch.Tensor:
    """Binary focal loss matching src/training/base_trainers.py."""
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean()


def assert_safe_path(path: Path, project_root: Path):
    """
    Strict safety assertion: ensures GAT outputs are NEVER written to the stable
    GraphSAGE models/ or checkpoints/ root directories.
    """
    resolved = Path(path).resolve()
    root_resolved = Path(project_root).resolve()

    forbidden_dirs = [
        (root_resolved / "models").resolve(),
        (root_resolved / "checkpoints").resolve(),
    ]
    for fb in forbidden_dirs:
        if resolved == fb or fb in resolved.parents:
            if "ESMGAT" not in resolved.parts:
                raise RuntimeError(
                    f"CRITICAL SAFETY VIOLATION: Attempted to write GAT artifact to forbidden path: {resolved}.\n"
                    f"All GAT outputs MUST be saved inside {root_resolved / 'ESMGAT'}."
                )


def _save_ckpt(path: Path, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
               scheduler: Any, best_state: Dict, best_loss: float, no_improve: int, project_root: Path):
    if path is None:
        return
    assert_safe_path(Path(path), project_root)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_state": best_state,
        "best_loss": best_loss,
        "no_improve": no_improve,
    }, tmp)
    os.replace(tmp, path)


def _try_resume(path: Path, model: nn.Module, optimizer: optim.Optimizer, scheduler: Any, device: torch.device):
    if path is None or not os.path.exists(path):
        return 0, None, float("inf"), 0
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        print(f"    Resumed GAT checkpoint from {path} at epoch {ck['epoch'] + 1}")
        return ck["epoch"] + 1, ck["best_state"], ck["best_loss"], ck["no_improve"]
    except Exception as e:
        print(f"    Could not resume GAT checkpoint ({e}); starting fresh.")
        return 0, None, float("inf"), 0


def fit_gat(
    model: nn.Module,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    src_fit: np.ndarray,
    dst_fit: np.ndarray,
    y_fit: np.ndarray,
    src_es: np.ndarray,
    dst_es: np.ndarray,
    y_es: np.ndarray,
    device: torch.device,
    project_root: Path,
    cfg: Dict[str, Any] = GAT_CFG,
    ckpt_path: str = None,
    ckpt_every: int = 5,
    tag: str = "GAT",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Full-batch GAT link-prediction training using the exact same two-stage detached-gradient
    routine as GraphSAGE:
      1. Encode all nodes to intermediate representation z (256-d).
      2. Detach z into z_det requiring grad.
      3. Decode training pairs in chunks (chunk_size=1000) accumulating loss into z_det.grad.
      4. Single backward pass through the GAT encoder: z.backward(z_det.grad).
      5. AdamW + CosineAnnealingLR + gradient clipping + early stopping on validation BCE loss.
    """
    model.to(device)
    x, ei = graph_x.to(device), graph_edge_index.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["eta_min"])

    s_f, d_f = torch.as_tensor(src_fit, device=device), torch.as_tensor(dst_fit, device=device)
    y_f = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    s_e, d_e = torch.as_tensor(src_es, device=device), torch.as_tensor(dst_es, device=device)
    y_e = torch.as_tensor(y_es, dtype=torch.float32, device=device)
    n, cs = len(s_f), cfg["chunk_size"]

    start, best_state, best_loss, no_improve = _try_resume(ckpt_path, model, opt, sched, device)
    best_epoch = start
    early_stopped = False
    history = []

    print(f"[{tag}] Training started for max {cfg['max_epochs']} epochs on {device} (patience={cfg['patience']})...")

    for epoch in range(start, cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        opt.zero_grad()

        # Step 1: Full-graph encode
        z = model.encode(x, ei)

        # Step 2: Detach intermediate embeddings
        z_det = z.detach().requires_grad_(True)
        tot_loss = 0.0

        # Step 3: Chunked pair decoding
        for i in range(0, n, cs):
            out = model.decode(z_det, s_f[i:i + cs], d_f[i:i + cs])
            loss = focal_loss(out.squeeze(1), y_f[i:i + cs], cfg["focal_alpha"], cfg["focal_gamma"]) * cfg["pair_weight"]
            (loss * cfg["loss_scale"]).backward()
            tot_loss += loss.item() * (len(out) / n)

        # Step 4: Backpropagate through GAT encoder
        z.backward(z_det.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        # Validation (Early stopping on validation BCE loss)
        model.eval()
        with torch.no_grad():
            zv = model.encode(x, ei)
            outs = [
                model.decode(zv, s_e[i:i + EVAL_CHUNK], d_e[i:i + EVAL_CHUNK])
                for i in range(0, len(s_e), EVAL_CHUNK)
            ]
            val_loss = F.binary_cross_entropy_with_logits(torch.cat(outs).squeeze(1), y_e).item()

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            no_improve += 1

        elapsed = time.time() - t0
        mem_info = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1e9
            mem_info = f" | VRAM {alloc:.2f}GB"

        print(
            f"  [{tag}] Epoch {epoch + 1:03d}/{cfg['max_epochs']} | Train Loss: {tot_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} {'*' if improved else ' '} | LR: {sched.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s{mem_info}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": tot_loss,
            "val_loss": val_loss,
            "improved": improved,
        })

        if (epoch + 1) % ckpt_every == 0 and ckpt_path:
            _save_ckpt(Path(ckpt_path), epoch, model, opt, sched, best_state, best_loss, no_improve, project_root)

        if no_improve >= cfg["patience"]:
            print(f"  [{tag}] Early stopping triggered at epoch {epoch + 1} (patience={cfg['patience']}).")
            early_stopped = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "early_stopped": early_stopped,
        "total_epochs_run": len(history),
        "history": history,
    }
    return model, summary


@torch.no_grad()
def gat_logits(
    model: nn.Module,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    src: np.ndarray,
    dst: np.ndarray,
    device: torch.device,
    chunk_size: int = EVAL_CHUNK,
) -> np.ndarray:
    """Extracts raw GAT logits for link pairs (src[i], dst[i])."""
    model.eval()
    z = model.encode(graph_x.to(device), graph_edge_index.to(device))
    s, d = torch.as_tensor(src, device=device), torch.as_tensor(dst, device=device)
    outs = [
        model.decode(z, s[i:i + chunk_size], d[i:i + chunk_size]).squeeze(1).cpu()
        for i in range(0, len(s), chunk_size)
    ]
    return torch.cat(outs).numpy()
