"""
Shared training routines and hyper-parameters for the two base models.

The OOF fold models in train_ensemble.py and the production models trained by train_sequence_model.py /
train_graph_model.py read the SAME constants below (max epochs, patience, LR, weight decay, scheduler, loss),
so an out-of-fold prediction comes from a model trained under the same recipe as the production model.

Early stopping needs a validation set. Production models use val.csv; each OOF fold carves a stratified
inner validation split out of the fold's TRAINING rows. The held-out OOF rows are never used for early stopping.
"""
import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SEQ_CFG = dict(max_epochs=30, patience=10, lr=1e-3, weight_decay=1e-3, eta_min=1e-5,
               focal_alpha=0.5, focal_gamma=2.0, noise_std=0.01, batch_size=128)
GRAPH_CFG = dict(max_epochs=100, patience=15, lr=1e-3, weight_decay=1e-4, eta_min=1e-6,
                 focal_alpha=0.3, focal_gamma=2.5, chunk_size=1000, loss_scale=10.0, pair_weight=0.5)
EVAL_CHUNK = 2000


def get_device(force_cpu: bool = False) -> torch.device:
    """CUDA when the runtime provides it (Colab T4), otherwise CPU."""
    return torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")


def focal_loss(inputs, targets, alpha: float = 0.5, gamma: float = 2.0):
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean()


def _save_ckpt(path, epoch, model, optimizer, scheduler, best_state, best_loss, no_improve):
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "best_state": best_state, "best_loss": best_loss,
                "no_improve": no_improve}, tmp)
    os.replace(tmp, path)  # atomic: a session killed mid-write cannot leave a truncated checkpoint


def _try_resume(path, model, optimizer, scheduler, device):
    if path is None or not os.path.exists(path):
        return 0, None, float("inf"), 0
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        print(f"    resumed from {path} at epoch {ck['epoch'] + 1}")
        return ck["epoch"] + 1, ck["best_state"], ck["best_loss"], ck["no_improve"]
    except Exception as e:  # incompatible / corrupt checkpoint -> train from scratch
        print(f"    could not resume ({e}); starting fresh")
        return 0, None, float("inf"), 0


def build_embedding_table(embeddings, proteins, device):
    """Mean-pooled float32 embedding per protein, stacked on `device`. Returns (table, protein->row)."""
    rows = []
    for p in proteins:
        e = embeddings[p].float()
        rows.append(e.mean(dim=0) if e.dim() > 1 else e)
    return torch.stack(rows).to(device), {p: i for i, p in enumerate(proteins)}


def fit_sequence(model, table, a_fit, b_fit, y_fit, a_es, b_es, y_es, device, cfg=SEQ_CFG,
                 ckpt_path=None, ckpt_every=5, tag="seq"):
    """
    Trains the sequence MLP on rows (table[a], table[b], y) with focal loss, AdamW, cosine annealing and early
    stopping on inner-validation focal loss. Returns the model loaded with its best-validation weights.
    """
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["eta_min"])
    a_fit, b_fit = torch.as_tensor(a_fit, device=device), torch.as_tensor(b_fit, device=device)
    y_fit = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    a_es, b_es = torch.as_tensor(a_es, device=device), torch.as_tensor(b_es, device=device)
    y_es = torch.as_tensor(y_es, dtype=torch.float32, device=device)
    n, bs = len(a_fit), cfg["batch_size"]

    start, best_state, best_loss, no_improve = _try_resume(ckpt_path, model, opt, sched, device)
    for epoch in range(start, cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            ids = perm[i:i + bs]
            if len(ids) < 2:  # BatchNorm1d needs >1 sample in train mode
                continue
            e1, e2 = table[a_fit[ids]], table[b_fit[ids]]
            flip = torch.rand(len(ids), 1, device=device) > 0.5
            e1, e2 = torch.where(flip, e2, e1), torch.where(flip, e1, e2)
            e1 = e1 + torch.randn_like(e1) * cfg["noise_std"]
            e2 = e2 + torch.randn_like(e2) * cfg["noise_std"]
            out = model(e1, e2)
            loss = focal_loss(out, y_fit[ids].unsqueeze(1), cfg["focal_alpha"], cfg["focal_gamma"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for i in range(0, len(a_es), 4096):
                out = model(table[a_es[i:i + 4096]], table[b_es[i:i + 4096]])
                vl += focal_loss(out, y_es[i:i + 4096].unsqueeze(1), cfg["focal_alpha"], cfg["focal_gamma"]).item()
                vb += 1
            val_loss = vl / max(vb, 1)
        improved = val_loss < best_loss
        if improved:
            best_loss, no_improve = val_loss, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            no_improve += 1
        print(f"    [{tag}] epoch {epoch + 1}/{cfg['max_epochs']} train {tot / max(nb, 1):.4f} val {val_loss:.4f} "
              f"{'*' if improved else ' '} ({time.time() - t0:.1f}s)")
        if (epoch + 1) % ckpt_every == 0:
            _save_ckpt(ckpt_path, epoch, model, opt, sched, best_state, best_loss, no_improve)
        if no_improve >= cfg["patience"]:
            print(f"    [{tag}] early stopping at epoch {epoch + 1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def fit_graph(model, graph_x, graph_edge_index, src_fit, dst_fit, y_fit, src_es, dst_es, y_es, device,
              cfg=GRAPH_CFG, ckpt_path=None, ckpt_every=5, tag="graph"):
    """
    Full-batch GraphSAGE link-prediction training (same scheme as train_graph_model.py): encode every node,
    decode supervision pairs in chunks against a detached z, then back-propagate the accumulated z-gradient.
    Early stopping on inner-validation BCE.
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
    for epoch in range(start, cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        z = model.encode(x, ei)
        z_det = z.detach().requires_grad_(True)
        tot = 0.0
        for i in range(0, n, cs):
            out = model.decode(z_det, s_f[i:i + cs], d_f[i:i + cs])
            loss = focal_loss(out.squeeze(1), y_f[i:i + cs], cfg["focal_alpha"], cfg["focal_gamma"]) * cfg["pair_weight"]
            (loss * cfg["loss_scale"]).backward()
            tot += loss.item() * (len(out) / n)
        z.backward(z_det.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            zv = model.encode(x, ei)
            outs = [model.decode(zv, s_e[i:i + EVAL_CHUNK], d_e[i:i + EVAL_CHUNK]) for i in range(0, len(s_e), EVAL_CHUNK)]
            val_loss = F.binary_cross_entropy_with_logits(torch.cat(outs).squeeze(1), y_e).item()
        improved = val_loss < best_loss
        if improved:
            best_loss, no_improve = val_loss, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            no_improve += 1
        print(f"    [{tag}] epoch {epoch + 1}/{cfg['max_epochs']} train {tot:.4f} val {val_loss:.4f} "
              f"{'*' if improved else ' '} ({time.time() - t0:.1f}s)")
        if (epoch + 1) % ckpt_every == 0:
            _save_ckpt(ckpt_path, epoch, model, opt, sched, best_state, best_loss, no_improve)
        if no_improve >= cfg["patience"]:
            print(f"    [{tag}] early stopping at epoch {epoch + 1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


@torch.no_grad()
def graph_logits(model, graph_x, graph_edge_index, src, dst, device, chunk_size=EVAL_CHUNK):
    """Raw GraphSAGE logits for the pairs (src[i], dst[i]); everything moved to `device`."""
    model.eval()
    z = model.encode(graph_x.to(device), graph_edge_index.to(device))
    s, d = torch.as_tensor(src, device=device), torch.as_tensor(dst, device=device)
    outs = [model.decode(z, s[i:i + chunk_size], d[i:i + chunk_size]).squeeze(1).cpu() for i in range(0, len(s), chunk_size)]
    return torch.cat(outs).numpy()


@torch.no_grad()
def sequence_probs(model, table, a, b, device, batch_size=4096):
    model.eval()
    a, b = torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
    outs = [torch.sigmoid(model(table[a[i:i + batch_size]], table[b[i:i + batch_size]])).squeeze(1).cpu()
            for i in range(0, len(a), batch_size)]
    return torch.cat(outs).numpy()
