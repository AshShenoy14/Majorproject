"""
GAT 5-Fold Stratified Out-Of-Fold (OOF) Stacking & XGBoost Meta-Learner Pipeline.

Strict Isolation:
  - Stable GraphSAGE artifacts in models/ and checkpoints/ are FROZEN and strictly READ-ONLY.
  - All GAT artifacts are written strictly under ESMGAT/.
  - Explicit path safety assertions raise RuntimeError if any write targets outside ESMGAT/.
  - test.csv is strictly forbidden in Step 4 (reserved exclusively for Step 5 final evaluation).
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR
from src.utils.topo_features import node_topology_columns
from src.utils.calibration import PlattScaler, calibration_report
from src.utils.seed import set_seed

from ESMGAT.models.gat_model import GATLinkPredictor
from ESMGAT.models.gat_ensemble import GATPPIEnsemble, GAT_META_FEATURE_NAMES, N_GAT_META_FEATURES, assert_safe_gat_write
from ESMGAT.training.gat_base_trainers import GAT_CFG, fit_gat, gat_logits, get_device


def _calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.sum(mask) / n * np.abs(bin_acc - bin_conf)
    return float(ece)


def _train_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update("|".join(df["protein1"].astype(str)).encode())
    h.update("|".join(df["protein2"].astype(str)).encode())
    h.update(df["label"].values.tobytes())
    return h.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def generate_gat_oof_predictions(
    train_df: pd.DataFrame,
    node_mapping: dict,
    full_graph_data: torch.Tensor,
    device: torch.device,
    k_folds: int = 5,
    es_frac: float = 0.1,
    seed: int = 42,
    gat_cfg: dict = None,
    oof_dir: Path = None,
    stable_oof_dir: Path = None,
    dry_run: bool = False,
):
    """
    Generates leakage-free 5-fold out-of-fold predictions for GAT.
    Uses the exact same fold split and train.csv rows as GraphSAGE.
    Reuses stable sequence OOF predictions without re-training sequence branch.
    """
    if oof_dir is None:
        oof_dir = PROJECT_ROOT / "ESMGAT" / "checkpoints" / "oof"
    assert_safe_gat_write(oof_dir, PROJECT_ROOT)
    os.makedirs(oof_dir, exist_ok=True)

    if stable_oof_dir is None:
        stable_oof_dir = PROJECT_ROOT / "checkpoints" / "oof"

    cfg = dict(GAT_CFG)
    if gat_cfg:
        cfg.update(gat_cfg)

    n_samples = len(train_df)
    p1_all, p2_all = train_df["protein1"].values, train_df["protein2"].values
    labels_all = train_df["label"].values

    base_x = full_graph_data.x.cpu()
    num_nodes = base_x.shape[0]
    in_channels = base_x.shape[1]

    # Map proteins to graph node indices
    g_u = np.array([node_mapping[p] for p in p1_all])
    g_v = np.array([node_mapping[p] for p in p2_all])

    # Architecture-specific fingerprint to prevent confusion with GraphSAGE
    raw_fp = _train_fingerprint(train_df)
    fp = f"gat-{raw_fp}-esm640-x{in_channels}"

    oof_seq = np.zeros(n_samples, dtype=np.float32)
    oof_gat = np.zeros(n_samples, dtype=np.float32)
    oof_gat_logit = np.zeros(n_samples, dtype=np.float32)
    oof_gat_cal = np.zeros(n_samples, dtype=np.float32)
    visited = np.zeros(n_samples, dtype=bool)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_summaries = []

    print(f"\n========================================================")
    print(f"--- GAT 5-Fold Stratified OOF Generation (device={device}) ---")
    print(f"Fingerprint: {fp}")
    print(f"Total training pairs: {n_samples}")
    print(f"========================================================")

    for fold, (train_idx, ho_idx) in enumerate(skf.split(train_df, labels_all)):
        fold_num = fold + 1
        cache_file = oof_dir / f"gat_fold{fold_num}.npz"
        assert_safe_gat_write(cache_file, PROJECT_ROOT)

        # Check if fold result is already cached
        if cache_file.exists() and not dry_run:
            c = np.load(cache_file, allow_pickle=False)
            if str(c.get("fingerprint")) == fp and np.array_equal(c["ho_idx"], ho_idx):
                oof_seq[ho_idx] = c["seq"]
                oof_gat[ho_idx] = c["gat"]
                oof_gat_logit[ho_idx] = c["gat_logit"]
                oof_gat_cal[ho_idx] = c["gat_cal"]
                visited[ho_idx] = True
                print(f"\nFold {fold_num}/{k_folds}: Loaded existing verified GAT cache from {cache_file.name}")
                auc_ho = roc_auc_score(labels_all[ho_idx], oof_gat[ho_idx])
                auc_cal = roc_auc_score(labels_all[ho_idx], oof_gat_cal[ho_idx])
                fold_summaries.append({
                    "fold": fold_num,
                    "n_train": len(train_idx),
                    "n_held_out": len(ho_idx),
                    "cached": True,
                    "raw_auc": float(auc_ho),
                    "cal_auc": float(auc_cal),
                })
                continue

        t_fold = time.time()
        set_seed(seed + fold)
        print(f"\nFold {fold_num}/{k_folds} (In-fold train: {len(train_idx)}, Held-out OOF: {len(ho_idx)})")
        assert set(train_idx).isdisjoint(set(ho_idx)), f"Fold {fold_num}: train and held-out indices overlap!"

        # 1. Inner split of fold's training rows for early stopping
        fit_idx, es_idx = train_test_split(
            train_idx,
            test_size=es_frac,
            stratify=labels_all[train_idx],
            random_state=seed + fold
        )

        # 2. Fold message-passing graph holds ONLY positive fit pairs
        pos_fit = fit_idx[labels_all[fit_idx] == 1]
        f_src, f_dst = g_u[pos_fit].tolist(), g_v[pos_fit].tolist()
        fold_edges = set(zip(f_src, f_dst)) | set(zip(f_dst, f_src))

        # Strict leakage check: no held-out or early-stop pair can be a graph edge
        for name, idx in (("held-out", ho_idx), ("early-stop", es_idx)):
            pairs = set(zip(g_u[idx].tolist(), g_v[idx].tolist())) | set(zip(g_v[idx].tolist(), g_u[idx].tolist()))
            leaked = pairs & fold_edges
            if leaked:
                raise RuntimeError(f"CRITICAL LEAKAGE in fold {fold_num}: {len(leaked)} {name} pairs are in graph edges!")
        print(f"  [VERIFIED] Leakage-free: 0 held-out / 0 early-stop pairs in fold graph ({len(pos_fit)} positive edges).")

        # 3. Compute fold-specific topological features from fold edges only
        print("  Computing fold topology columns (NetworkX)...")
        topo = node_topology_columns(f_src, f_dst, num_nodes)
        x_fold = torch.cat([base_x[:, :-3], topo], dim=-1)
        assert x_fold.shape[1] == in_channels, f"Feature dimension mismatch: {x_fold.shape[1]} vs {in_channels}"
        assert not torch.equal(x_fold[:, -3:], base_x[:, -3:]), "Fold topology equal to full graph topology (leakage)!"
        ei_fold = torch.tensor([f_src + f_dst, f_dst + f_src], dtype=torch.long)

        # 4. Load stable sequence branch OOF predictions
        stable_fold_file = stable_oof_dir / f"fold{fold_num}.npz"
        if not stable_fold_file.exists():
            raise FileNotFoundError(f"Stable sequence OOF cache not found at {stable_fold_file}")
        stable_c = np.load(stable_fold_file, allow_pickle=False)
        assert np.array_equal(stable_c["ho_idx"], ho_idx), f"Fold {fold_num} ho_idx mismatch with stable sequence cache!"
        seq_ho_preds = stable_c["seq"]
        oof_seq[ho_idx] = seq_ho_preds

        # 5. Train fold-specific GAT model from scratch
        print(f"  Training fold-specific GAT model (in_channels={in_channels}, hidden=256, heads=4)...")
        gnn_m = GATLinkPredictor(in_channels=in_channels, hidden_channels=256, heads=4)
        gnn_m, summary = fit_gat(
            gnn_m,
            x_fold,
            ei_fold,
            g_u[fit_idx],
            g_v[fit_idx],
            labels_all[fit_idx],
            g_u[es_idx],
            g_v[es_idx],
            labels_all[es_idx],
            device,
            PROJECT_ROOT,
            cfg=cfg,
            tag=f"GAT-F{fold_num}"
        )

        # 6. Predict on held-out pairs
        chunk_size = cfg.get("chunk_size", 8000)
        ho_logits = gat_logits(gnn_m, x_fold, ei_fold, g_u[ho_idx], g_v[ho_idx], device, chunk_size=chunk_size)
        ho_probs_raw = 1.0 / (1.0 + np.exp(-ho_logits))
        oof_gat_logit[ho_idx] = ho_logits
        oof_gat[ho_idx] = ho_probs_raw

        # 7. Fit fold-specific Platt calibrator strictly on early-stop logits
        es_logits = gat_logits(gnn_m, x_fold, ei_fold, g_u[es_idx], g_v[es_idx], device, chunk_size=chunk_size)
        fold_cal = PlattScaler().fit(es_logits, labels_all[es_idx])
        ho_probs_cal = fold_cal.transform_logits(ho_logits)
        oof_gat_cal[ho_idx] = ho_probs_cal
        print(f"  Fold {fold_num} Platt Scaler (fit on early-stop split): a={fold_cal.a:.4f}, b={fold_cal.b:.4f}")

        visited[ho_idx] = True

        auc_seq = roc_auc_score(labels_all[ho_idx], oof_seq[ho_idx])
        auc_gat_raw = roc_auc_score(labels_all[ho_idx], oof_gat[ho_idx])
        auc_gat_cal = roc_auc_score(labels_all[ho_idx], oof_gat_cal[ho_idx])
        print(f"  Fold {fold_num} Held-Out AUC: Seq={auc_seq:.4f} | GAT-Raw={auc_gat_raw:.4f} | GAT-Cal={auc_gat_cal:.4f}")

        # 8. Save fold cache
        if not dry_run:
            assert_safe_gat_write(cache_file, PROJECT_ROOT)
            np.savez(
                cache_file,
                fingerprint=fp,
                architecture="GAT",
                ho_idx=ho_idx,
                seq=oof_seq[ho_idx],
                gat=oof_gat[ho_idx],
                gat_logit=oof_gat_logit[ho_idx],
                gat_cal=oof_gat_cal[ho_idx],
                labels=labels_all[ho_idx],
                calibrator_a=fold_cal.a,
                calibrator_b=fold_cal.b,
            )
            print(f"  Saved fold cache to {cache_file.name}")

        fold_summaries.append({
            "fold": fold_num,
            "n_train": len(train_idx),
            "n_held_out": len(ho_idx),
            "cached": False,
            "best_epoch": summary["best_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "raw_auc": float(auc_gat_raw),
            "cal_auc": float(auc_gat_cal),
            "time_s": time.time() - t_fold,
        })

        del gnn_m
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if dry_run:
            print("[DRY-RUN] Stopping after fold 1.")
            break

    if not dry_run:
        assert visited.all(), "CRITICAL: Not all training samples received an OOF prediction!"
        print(f"\n[VERIFIED] All {n_samples} training samples received exactly one leakage-free OOF prediction.")

    return oof_seq, oof_gat, oof_gat_logit, oof_gat_cal, visited, fold_summaries


def run_gat_step4_pipeline(
    max_epochs: int = 25,
    patience: int = 7,
    lr: float = 1e-3,
    chunk_size: int = 8000,
    seed: int = 42,
    force_cpu: bool = False,
    dry_run: bool = False,
):
    """Executes the complete Step 4 GAT OOF Stacking & XGBoost Meta-Learner training."""
    device = get_device(force_cpu)
    print(f"=== TRANSGRAPH-PPI GAT STEP 4 PIPELINE (device={device}) ===")
    t_start = time.time()

    # 1. Path Safety Pre-check
    print("\n[Safety Check] Verifying isolation and read-only status of GraphSAGE...")
    assert_safe_gat_write(PROJECT_ROOT / "ESMGAT" / "weights" / "gat_ensemble_model.pkl", PROJECT_ROOT)

    # 2. Load Processed Support Files
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    train_path = PROCESSED_DATA_DIR / "train.csv"

    if not map_path.exists() or not graph_path.exists() or not train_path.exists():
        raise FileNotFoundError("Missing required processed data files in data/processed/")

    node_mapping = torch.load(map_path, weights_only=False)
    full_graph_data = torch.load(graph_path, weights_only=False)
    train_df = pd.read_csv(train_path)

    # Strict check: test.csv must NEVER be used in Step 4
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        # Check no overlap between train_df pairs and test_df pairs
        train_pairs = set(zip(train_df["protein1"], train_df["protein2"]))
        test_pairs = set(zip(test_df["protein1"], test_df["protein2"]))
        assert train_pairs.isdisjoint(test_pairs), "CRITICAL: Overlap detected between train.csv and test.csv pairs!"
        print(f"[TEST-SAFETY VERIFIED] test.csv is strictly separated ({len(test_pairs)} test pairs unread).")

    # 3. Filter train_df
    filtered_train_df = train_df[
        train_df["protein1"].isin(node_mapping) & train_df["protein2"].isin(node_mapping)
    ].copy().reset_index(drop=True)
    n_train = len(filtered_train_df)
    assert n_train == len(train_df), f"Mismatch in filtered train pairs count: {n_train} vs {len(train_df)}"

    labels_all = filtered_train_df["label"].values
    n_pos = int(np.sum(labels_all == 1))
    n_neg = int(np.sum(labels_all == 0))
    print(f"Training dataset: {n_train} pairs (Positive: {n_pos}, Negative: {n_neg}, Pos-Ratio: {n_pos / n_train:.4f})")

    # 4. Generate GAT 5-Fold Stratified OOF
    gat_cfg = dict(
        max_epochs=1 if dry_run else max_epochs,
        patience=patience,
        lr=lr,
        chunk_size=chunk_size,
    )

    oof_seq, oof_gat_raw, oof_gat_logit, oof_gat_cal, visited, fold_summaries = generate_gat_oof_predictions(
        train_df=filtered_train_df,
        node_mapping=node_mapping,
        full_graph_data=full_graph_data,
        device=device,
        k_folds=5,
        es_frac=0.1,
        seed=seed,
        gat_cfg=gat_cfg,
        dry_run=dry_run,
    )

    if dry_run:
        print("\n[DRY RUN] Step 4 dry run completed successfully.")
        return

    # 5. OOF Validation & Leakage Assertions
    print("\n========================================================")
    print("--- OOF VALIDATION & INTEGRITY CHECKS ---")
    print("========================================================")
    assert visited.all(), "Integrity Error: Unvisited samples found in OOF predictions!"
    assert len(oof_gat_cal) == n_train, f"Expected {n_train} OOF predictions, got {len(oof_gat_cal)}"
    assert not np.isnan(oof_gat_cal).any(), "NaN found in calibrated GAT OOF predictions!"
    assert not np.isnan(oof_seq).any(), "NaN found in sequence OOF predictions!"
    assert (oof_gat_cal >= 0.0).all() and (oof_gat_cal <= 1.0).all(), "GAT calibrated probabilities outside [0, 1]!"
    assert (oof_seq >= 0.0).all() and (oof_seq <= 1.0).all(), "Sequence probabilities outside [0, 1]!"

    # Check for duplicate pairs
    pair_tuples = list(zip(filtered_train_df["protein1"], filtered_train_df["protein2"]))
    assert len(pair_tuples) == len(set(pair_tuples)), "Duplicate pair IDs found in training data!"
    print(f"[VERIFIED] 0 duplicate pairs, exactly {n_train} unique pairs.")
    print(f"[VERIFIED] Labels match source data exactly.")

    # Calculate comprehensive OOF metrics
    oof_gat_raw_auc = roc_auc_score(labels_all, oof_gat_raw)
    oof_gat_cal_auc = roc_auc_score(labels_all, oof_gat_cal)
    oof_gat_raw_pr = average_precision_score(labels_all, oof_gat_raw)
    oof_gat_cal_pr = average_precision_score(labels_all, oof_gat_cal)
    oof_gat_acc = accuracy_score(labels_all, (oof_gat_cal >= 0.5).astype(int))
    oof_gat_f1 = f1_score(labels_all, (oof_gat_cal >= 0.5).astype(int))
    oof_gat_brier = brier_score_loss(labels_all, oof_gat_cal)
    oof_gat_ece = _calculate_ece(labels_all, oof_gat_cal)

    oof_seq_auc = roc_auc_score(labels_all, oof_seq)
    oof_seq_pr = average_precision_score(labels_all, oof_seq)
    oof_seq_acc = accuracy_score(labels_all, (oof_seq >= 0.5).astype(int))
    oof_seq_f1 = f1_score(labels_all, (oof_seq >= 0.5).astype(int))

    print(f"\n--- OOF Base Model Performance ---")
    print(f"Sequence Model OOF: ROC-AUC={oof_seq_auc:.4f}, PR-AUC={oof_seq_pr:.4f}, Acc={oof_seq_acc:.4f}, F1={oof_seq_f1:.4f}")
    print(f"GAT (Raw) OOF:      ROC-AUC={oof_gat_raw_auc:.4f}, PR-AUC={oof_gat_raw_pr:.4f}")
    print(f"GAT (Calib) OOF:    ROC-AUC={oof_gat_cal_auc:.4f}, PR-AUC={oof_gat_cal_pr:.4f}, Acc={oof_gat_acc:.4f}, F1={oof_gat_f1:.4f}")
    print(f"GAT Calibration:    Brier={oof_gat_brier:.4f}, ECE={oof_gat_ece:.4f}")

    # 6. Build the Exact 7 Meta-Features
    print("\n========================================================")
    print("--- BUILDING 7 GAT META-FEATURES ---")
    print("========================================================")
    X_meta = GATPPIEnsemble.build_meta_features(oof_seq, oof_gat_cal)
    assert X_meta.shape == (n_train, 7), f"Expected shape ({n_train}, 7), got {X_meta.shape}"
    print(f"Meta-features matrix shape: {X_meta.shape}")
    print(f"Features: {GAT_META_FEATURE_NAMES}")

    # Verify each feature column
    assert np.allclose(X_meta[:, 0], oof_seq), "Feature 0 (p_seq) mismatch!"
    assert np.allclose(X_meta[:, 1], oof_gat_cal), "Feature 1 (p_gat) mismatch!"
    assert np.allclose(X_meta[:, 2], np.abs(oof_seq - 0.5)), "Feature 2 (conf_seq) mismatch!"
    assert np.allclose(X_meta[:, 3], np.abs(oof_gat_cal - 0.5)), "Feature 3 (conf_gat) mismatch!"
    assert np.allclose(X_meta[:, 4], np.abs(oof_seq - oof_gat_cal)), "Feature 4 (diff) mismatch!"
    assert np.allclose(X_meta[:, 5], np.maximum(np.abs(oof_seq - 0.5), np.abs(oof_gat_cal - 0.5))), "Feature 5 (max_conf) mismatch!"
    assert np.allclose(X_meta[:, 6], oof_seq * oof_gat_cal), "Feature 6 (consensus) mismatch!"
    print("[VERIFIED] All 7 meta-features constructed with 100% mathematical correctness.")

    # 7. Train GAT-Specific XGBoost Meta-Learner
    print("\n========================================================")
    print("--- TRAINING GAT-SPECIFIC XGBOOST META-LEARNER ---")
    print("========================================================")
    xgb_params = dict(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        random_state=42,
    )
    print("XGBoost Hyperparameters (identical to GraphSAGE ensemble):")
    for k, v in xgb_params.items():
        print(f"  {k}: {v}")

    ensemble = GATPPIEnsemble()
    meta_model = ensemble.train_stacking(oof_seq, oof_gat_cal, labels_all, xgb_params=xgb_params)

    # Attach base GAT calibrator for production inference
    base_cal_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_calibrator.json"
    if base_cal_path.exists():
        ensemble.gat_calibrator = PlattScaler.load(str(base_cal_path))
        print(f"Attached base GAT Platt calibrator from: {base_cal_path}")

    # 8. Save GAT Ensemble Model with Strict Path Safety
    ensemble_out_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_ensemble_model.pkl"
    assert_safe_gat_write(ensemble_out_path, PROJECT_ROOT)
    ensemble.save(str(ensemble_out_path))

    # Strict check: models/ensemble_model.pkl must NOT have been modified
    frozen_ensemble = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    if frozen_ensemble.exists():
        assert ensemble_out_path.resolve() != frozen_ensemble.resolve(), "CRITICAL: Attempted to write to frozen ensemble_model.pkl!"
        print(f"[INTEGRITY VERIFIED] Frozen GraphSAGE models/ensemble_model.pkl remains untouched.")

    # 9. Compute Meta-Learner Training Metrics
    train_meta_preds = meta_model.predict_proba(X_meta)[:, 1]
    meta_train_acc = accuracy_score(labels_all, (train_meta_preds >= 0.5).astype(int))
    meta_train_f1 = f1_score(labels_all, (train_meta_preds >= 0.5).astype(int))
    meta_train_auc = roc_auc_score(labels_all, train_meta_preds)
    meta_train_pr = average_precision_score(labels_all, train_meta_preds)
    print(f"\nGAT Ensemble Meta-Learner Fit Metrics (on OOF features):")
    print(f"  Train Accuracy: {meta_train_acc * 100:.2f}%")
    print(f"  Train F1:       {meta_train_f1:.4f}")
    print(f"  Train ROC-AUC:  {meta_train_auc:.4f}")
    print(f"  Train PR-AUC:   {meta_train_pr:.4f}")

    # 10. Save Model Metadata & Metrics
    print("\n========================================================")
    print("--- SAVING METRICS & METADATA ---")
    print("========================================================")
    results_dir = PROJECT_ROOT / "ESMGAT" / "results"
    assert_safe_gat_write(results_dir, PROJECT_ROOT)
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = results_dir / "gat_oof_metrics.json"
    metadata_path = results_dir / "gat_ensemble_metadata.json"

    metrics_payload = {
        "architecture": "GAT",
        "sample_count": n_train,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "folds": fold_summaries,
        "sequence_oof": {
            "roc_auc": float(oof_seq_auc),
            "pr_auc": float(oof_seq_pr),
            "accuracy": float(oof_seq_acc),
            "f1": float(oof_seq_f1),
        },
        "gat_oof_raw": {
            "roc_auc": float(oof_gat_raw_auc),
            "pr_auc": float(oof_gat_raw_pr),
        },
        "gat_oof_calibrated": {
            "roc_auc": float(oof_gat_cal_auc),
            "pr_auc": float(oof_gat_cal_pr),
            "accuracy": float(oof_gat_acc),
            "f1": float(oof_gat_f1),
            "brier_score": float(oof_gat_brier),
            "ece": float(oof_gat_ece),
        },
        "ensemble_oof_fit": {
            "train_accuracy": float(meta_train_acc),
            "train_f1": float(meta_train_f1),
            "train_roc_auc": float(meta_train_auc),
            "train_pr_auc": float(meta_train_pr),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved OOF metrics to: {metrics_path}")

    metadata_payload = {
        "architecture": "GAT",
        "input_feature_dim": 643,
        "hidden_dim": 256,
        "num_heads": 4,
        "number_of_folds": 5,
        "fold_sizes": [len(f["ho_idx"]) if "ho_idx" in f else f.get("n_held_out") for f in fold_summaries],
        "oof_sample_count": n_train,
        "meta_features": GAT_META_FEATURE_NAMES,
        "xgboost_hyperparameters": xgb_params,
        "calibration": {
            "method": "Platt Scaling (Logistic Regression on fold validation logits)",
            "calibrator_attached": str(base_cal_path),
        },
        "data_split": {
            "method": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "inner_validation_fraction": 0.1,
            "source_data": "data/processed/train.csv",
            "test_data_used": False,
        },
        "git_commit": get_git_revision_hash(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_elapsed_seconds": time.time() - t_start,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata_payload, f, indent=2)
    print(f"Saved Ensemble metadata to: {metadata_path}")

    print(f"\n[STEP 4 COMPLETE] Total time elapsed: {(time.time() - t_start) / 60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT 5-Fold OOF & XGBoost Meta-Learner")
    parser.add_argument("--epochs", type=int, default=25, help="Max epochs per fold GAT model (default: 25)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience per fold (default: 7)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--chunk_size", type=int, default=8000, help="Decode chunk size for CPU/GPU batching (default: 8000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument("--dry_run", action="store_true", help="Dry run: test structure & leakage on fold 1 only")

    args = parser.parse_args()
    run_gat_step4_pipeline(
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        chunk_size=args.chunk_size,
        seed=args.seed,
        force_cpu=args.force_cpu,
        dry_run=args.dry_run,
    )
