import numpy as np
import torch

from src.utils.calibration import PlattScaler, ece_score, calibration_report, cross_fitted_probs
from src.utils.topo_features import node_topology_columns
from src.training.base_trainers import build_embedding_table, fit_sequence, SEQ_CFG
from src.models.sequence_model import SequencePPIModel
from src.utils.esm_config import ESM_EMBED_DIM


def test_platt_scaling_reduces_ece_on_overconfident_scores():
    rng = np.random.RandomState(0)
    true_p = rng.rand(6000)
    y = (rng.rand(6000) < true_p).astype(int)
    logits = 4.0 * np.log(true_p / (1 - true_p))          # over-confident logits
    p_raw = 1 / (1 + np.exp(-logits))
    scaler = PlattScaler().fit(logits, y)
    rep = calibration_report(y, p_raw, scaler.transform_logits(logits))
    assert rep["after"]["ECE"] < rep["before"]["ECE"] / 3
    assert abs(scaler.a - 0.25) < 0.05                      # recovers the 1/4 shrinkage
    assert ece_score(y, cross_fitted_probs(logits, y)) < rep["before"]["ECE"] / 3


def test_platt_scaler_roundtrip(tmp_path):
    PlattScaler(1.7, -0.3).save(tmp_path / "c.json")
    s = PlattScaler.load(tmp_path / "c.json")
    assert (s.a, s.b) == (1.7, -0.3)


def test_fold_topology_uses_only_fold_edges():
    """Node columns recomputed on a subgraph must differ from the full graph's and ignore held-out edges."""
    full_src, full_dst = [0, 1, 2, 3], [1, 2, 3, 0]
    fold_src, fold_dst = [0, 1], [1, 2]                     # edges (2,3),(3,0) held out
    full = node_topology_columns(full_src, full_dst, 4)
    fold = node_topology_columns(fold_src, fold_dst, 4)
    assert not torch.equal(full, fold)
    assert fold[3, 0] == 0.0 and fold[3, 2] > 0             # node 3 isolated in the fold: zero degree centrality
    assert full[3, 0] > 0


def test_fit_sequence_early_stops_and_resumes(tmp_path):
    torch.manual_seed(0)
    emb = {f"p{i}": torch.randn(ESM_EMBED_DIM) for i in range(40)}
    names = sorted(emb)
    table, row = build_embedding_table(emb, names, torch.device("cpu"))
    rng = np.random.RandomState(0)
    a, b = rng.randint(0, 40, 300), rng.randint(0, 40, 300)
    y = rng.randint(0, 2, 300)                              # pure noise -> validation loss cannot keep improving
    cfg = dict(SEQ_CFG, max_epochs=30, patience=2, batch_size=64)
    ck = str(tmp_path / "ck.pt")
    m = SequencePPIModel(input_dim=ESM_EMBED_DIM)
    fit_sequence(m, table, a[:240], b[:240], y[:240], a[240:], b[240:], y[240:], torch.device("cpu"),
                 cfg=cfg, ckpt_path=ck, ckpt_every=1, tag="t")
    assert (tmp_path / "ck.pt").exists()
    ckpt = torch.load(ck, weights_only=False)
    assert ckpt["epoch"] + 1 < 30                           # early stopping fired well before max epochs
    assert set(ckpt) >= {"model", "optimizer", "scheduler", "best_state", "best_loss", "no_improve"}
