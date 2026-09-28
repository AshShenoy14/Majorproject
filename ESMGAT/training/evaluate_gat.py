"""
Step 5: Final Held-Out Test Set Evaluation, GraphSAGE vs GAT Scientific Comparison,
and GAT Ensemble SHAP Analysis.

Strict Evaluation Constraints:
- Evaluation ONLY. Frozen artifacts in models/ and checkpoints/ are physically read-only.
- All generated artifacts are written strictly under ESMGAT/.
- test.csv is untouched and used strictly for held-out evaluation (zero threshold tuning on test).
- Pre-established validation thresholds are reused.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn.functional as F
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor
from src.utils.calibration import PlattScaler
from src.utils.paths import PROCESSED_DATA_DIR

from ESMGAT.models.gat_model import GATLinkPredictor
from ESMGAT.models.gat_ensemble import (
    GATPPIEnsemble,
    GAT_META_FEATURE_NAMES,
    N_GAT_META_FEATURES,
    assert_safe_gat_write,
)


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"CRITICAL SHA256 HASH MISMATCH on {path}!\n"
            f"Expected: {expected}\nActual:   {actual}"
        )
    return True


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    print("=" * 75)
    print(" STEP 5: FINAL TEST-SET EVALUATION & GAT SHAP ANALYSIS")
    print("=" * 75)
    device = torch.device("cpu")
    print(f"Evaluation execution device: {device}")

    # =========================================================================
    # 0. VERIFY FROZEN ARTIFACT INTEGRITY (SHA256)
    # =========================================================================
    print("\n[0/9] Verifying frozen GraphSAGE artifact integrity...")
    frozen_files = {
        PROJECT_ROOT / "models" / "graph_model_best.pth": "16d3b16ea464f5e1bd435c8c2bd3aac2cf2415a8f50fb03745f87bd4aa5d57fa",
        PROJECT_ROOT / "models" / "ensemble_model.pkl": "7e2c9543cb16f19429292b449128e03ace1bd88cc115c1db79fa29f1c8e98b60",
        PROJECT_ROOT / "models" / "graph_calibrator.json": "dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd",
    }
    for path, expected_hash in frozen_files.items():
        assert path.exists(), f"Missing frozen file: {path}"
        verify_sha256(path, expected_hash)
        print(f"  [OK] {path.name}: {expected_hash[:16]}...")

    # Also verify GAT artifacts exist under ESMGAT/
    gat_model_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_model_best.pth"
    gat_cal_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_calibrator.json"
    gat_ens_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_ensemble_model.pkl"
    seq_model_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"

    assert gat_model_path.exists(), f"Missing GAT model: {gat_model_path}"
    assert gat_cal_path.exists(), f"Missing GAT calibrator: {gat_cal_path}"
    assert gat_ens_path.exists(), f"Missing GAT ensemble: {gat_ens_path}"
    assert seq_model_path.exists(), f"Missing sequence model: {seq_model_path}"
    print("  [OK] All model artifacts verified present.")

    # =========================================================================
    # 1. TEST SET VERIFICATION & LEAKAGE AUDIT
    # =========================================================================
    print("\n[1/9] Loading and verifying test set isolation...")
    test_path = PROCESSED_DATA_DIR / "test.csv"
    train_path = PROCESSED_DATA_DIR / "train.csv"
    assert test_path.exists(), f"Missing test.csv: {test_path}"
    assert train_path.exists(), f"Missing train.csv: {train_path}"

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    n_test = len(test_df)
    print(f"  Test sample count: {n_test}")
    assert n_test == 20172, f"Expected exactly 20,172 test rows, found {n_test}"

    # Symmetric pair leakage audit
    train_pairs_sym = set()
    for p1, p2 in zip(train_df["protein1"], train_df["protein2"]):
        train_pairs_sym.add((p1, p2))
        train_pairs_sym.add((p2, p1))

    test_pairs = list(zip(test_df["protein1"], test_df["protein2"]))
    leakage_count = sum(1 for p in test_pairs if p in train_pairs_sym)
    print(f"  Pair overlap between train.csv ({len(train_df)}) and test.csv ({n_test}): {leakage_count}")
    assert leakage_count == 0, f"DATA LEAKAGE DETECTED: {leakage_count} test pairs found in train set!"
    print("  [LEAKAGE AUDIT PASSED] Zero pair overlap between training and held-out test sets.")

    # =========================================================================
    # 2. VERIFY ACTUAL INPUT DIMENSIONS
    # =========================================================================
    print("\n[2/9] Verifying actual model input dimensions from checkpoints...")
    seq_sd = torch.load(seq_model_path, map_location="cpu", weights_only=False)
    sage_sd = torch.load(PROJECT_ROOT / "models" / "graph_model_best.pth", map_location="cpu", weights_only=False)
    gat_sd = torch.load(gat_model_path, map_location="cpu", weights_only=False)

    seq_in_dim = seq_sd["input_proj.0.weight"].shape[1] // 4  # feature_dim is input_dim * 4
    sage_in_dim = sage_sd["input_norm.weight"].shape[0]
    gat_in_dim = gat_sd["input_norm.weight"].shape[0]

    print(f"  Sequence Model ESM input dimension: {seq_in_dim} (input_proj weight shape: {seq_sd['input_proj.0.weight'].shape})")
    print(f"  GraphSAGE input feature dimension:  {sage_in_dim} (input_norm shape: {sage_sd['input_norm.weight'].shape})")
    print(f"  GAT input feature dimension:        {gat_in_dim} (input_norm shape: {gat_sd['input_norm.weight'].shape})")

    assert seq_in_dim == 640, f"Expected 640 ESM dim for sequence model, got {seq_in_dim}"
    assert sage_in_dim == 643, f"Expected 643 features for GraphSAGE, got {sage_in_dim}"
    assert gat_in_dim == 643, f"Expected 643 features for GAT, got {gat_in_dim}"
    print("  [OK] Actual input dimensions verified.")

    # =========================================================================
    # 3. LOAD DATA & GRAPH METADATA
    # =========================================================================
    print("\n[3/9] Loading embeddings, node mapping, and STRING graph...")
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, map_location="cpu", weights_only=False)
    graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)

    print(f"  Graph node features shape: {graph_data.x.shape}")
    print(f"  Graph edge index shape:    {graph_data.edge_index.shape}")
    assert graph_data.x.shape[1] == 643, f"Graph features shape mismatch: expected 643, got {graph_data.x.shape[1]}"

    # Verify all test pairs exist in node mapping and embeddings
    for p1, p2 in test_pairs:
        assert p1 in node_mapping and p2 in node_mapping, f"Protein pair ({p1}, {p2}) missing in graph mapping!"
        assert p1 in embeddings and p2 in embeddings, f"Protein pair ({p1}, {p2}) missing in embeddings!"

    # =========================================================================
    # 4. RUN TEST SET INFERENCE
    # =========================================================================
    print("\n[4/9] Running test inference across all 5 models...")
    test_labels = test_df["label"].values.astype(int)

    # 4a. Sequence Model Inference
    print("  Running Model 1/5: ESM-MLP Sequence Model...")
    seq_model = SequencePPIModel(input_dim=640, hidden_dim=1024)
    seq_model.load_state_dict(seq_sd)
    seq_model.eval()

    p_seq_list = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            chunk = test_df.iloc[i : i + batch_size]
            e1 = torch.stack([embeddings[p] for p in chunk["protein1"]]).to(device)
            e2 = torch.stack([embeddings[p] for p in chunk["protein2"]]).to(device)
            out = seq_model(e1, e2)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            p_seq_list.extend(probs)
    p_seq = np.array(p_seq_list, dtype=np.float64)

    # 4b. Graph Models Inference
    g_src = [node_mapping[p1] for p1 in test_df["protein1"]]
    g_dst = [node_mapping[p2] for p2 in test_df["protein2"]]
    edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long)

    # GraphSAGE
    print("  Running Model 2/5: GraphSAGE Graph Model...")
    sage_model = SAGELinkPredictor(in_channels=643, hidden_channels=256)
    sage_model.load_state_dict(sage_sd)
    sage_model.eval()

    with torch.no_grad():
        z_sage = sage_model.encode(graph_data.x, graph_data.edge_index)
        p_sage_raw_list = []
        chunk_size = 5000
        for i in range(0, n_test, chunk_size):
            c_src = edge_label_index[0, i : i + chunk_size]
            c_dst = edge_label_index[1, i : i + chunk_size]
            out = sage_model.decode(z_sage, c_src, c_dst)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            p_sage_raw_list.extend(probs)
    p_graphsage_raw = np.array(p_sage_raw_list, dtype=np.float64)

    # GAT
    print("  Running Model 3/5: GAT Graph Model...")
    gat_model = GATLinkPredictor(in_channels=643, hidden_channels=256, heads=4, dropout=0.4)
    gat_model.load_state_dict(gat_sd)
    gat_model.eval()

    with torch.no_grad():
        z_gat = gat_model.encode(graph_data.x, graph_data.edge_index)
        p_gat_raw_list = []
        chunk_size = 5000
        for i in range(0, n_test, chunk_size):
            c_src = edge_label_index[0, i : i + chunk_size]
            c_dst = edge_label_index[1, i : i + chunk_size]
            out = gat_model.decode(z_gat, c_src, c_dst)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            p_gat_raw_list.extend(probs)
    p_gat_raw = np.array(p_gat_raw_list, dtype=np.float64)

    # 4c. Calibration
    print("  Applying model-specific Platt calibrators...")
    sage_calibrator = PlattScaler.load(PROJECT_ROOT / "models" / "graph_calibrator.json")
    gat_calibrator = PlattScaler.load(gat_cal_path)

    p_graphsage = sage_calibrator.transform_probs(p_graphsage_raw)
    p_gat = gat_calibrator.transform_probs(p_gat_raw)

    # 4d. Ensemble Inference (7 Meta-Features each)
    print("  Running Model 4/5: ESM + GraphSAGE + XGBoost Ensemble...")
    sage_ensemble_model = joblib.load(PROJECT_ROOT / "models" / "ensemble_model.pkl")
    X_sage_meta = GATPPIEnsemble.build_meta_features(p_seq, p_graphsage)
    p_graphsage_ensemble = sage_ensemble_model.predict_proba(X_sage_meta)[:, 1]

    print("  Running Model 5/5: ESM + GAT + XGBoost Ensemble...")
    gat_ensemble_model = joblib.load(gat_ens_path)
    X_gat_meta = GATPPIEnsemble.build_meta_features(p_seq, p_gat)
    p_gat_ensemble = gat_ensemble_model.predict_proba(X_gat_meta)[:, 1]

    # Verify probability integrity
    prob_arrays = {
        "p_seq": p_seq,
        "p_graphsage_raw": p_graphsage_raw,
        "p_graphsage": p_graphsage,
        "p_gat_raw": p_gat_raw,
        "p_gat": p_gat,
        "p_graphsage_ensemble": p_graphsage_ensemble,
        "p_gat_ensemble": p_gat_ensemble,
    }
    for name, arr in prob_arrays.items():
        assert len(arr) == n_test, f"{name} length mismatch: expected {n_test}, got {len(arr)}"
        assert not np.isnan(arr).any(), f"NaN values detected in {name}!"
        assert not np.isinf(arr).any(), f"Inf values detected in {name}!"
        assert (arr >= 0.0).all() and (arr <= 1.0).all(), f"{name} values outside [0, 1]!"
    print("  [INTEGRITY VERIFIED] All predictions have exactly 20,172 finite values in [0, 1].")

    # =========================================================================
    # 5. THRESHOLD POLICY & METRICS CALCULATION
    # =========================================================================
    print("\n[5/9] Applying threshold policy and computing metrics...")

    # Threshold provenance documentation:
    # 1. ESM-MLP: 0.45 (selected on val.csv via F1 sweep)
    # 2. GraphSAGE: 0.61 (selected on val.csv via F1 sweep)
    # 3. GAT: 0.50 (established during validation & OOF calibration evaluation)
    # 4. GraphSAGE Ensemble: 0.45 (selected on val.csv via F1 sweep)
    # 5. GAT Ensemble: 0.50 (established during OOF meta-learner training)

    model_configs = {
        "ESM-MLP": {
            "probs": p_seq,
            "threshold": 0.45,
            "threshold_source": "Validation F1-max sweep on val.csv (esm150m_results/assets/evaluation/final_test_metrics.json)",
            "calibrator": "None (direct sigmoid probabilities)",
            "artifact": "models/sequence_model_best.pth",
            "input_dim": 2560,
        },
        "GraphSAGE": {
            "probs": p_graphsage,
            "threshold": 0.61,
            "threshold_source": "Validation F1-max sweep on val.csv (esm150m_results/assets/evaluation/final_test_metrics.json)",
            "calibrator": "models/graph_calibrator.json (PlattScaler)",
            "artifact": "models/graph_model_best.pth",
            "input_dim": 643,
        },
        "GAT": {
            "probs": p_gat,
            "threshold": 0.50,
            "threshold_source": "Validation & OOF standard evaluation threshold (ESMGAT/training/train_gat_colab.py, ESMGAT/results/gat_val_metrics.json)",
            "calibrator": "ESMGAT/weights/gat_calibrator.json (PlattScaler)",
            "artifact": "ESMGAT/weights/gat_model_best.pth",
            "input_dim": 643,
        },
        "ESM + GraphSAGE + XGBoost": {
            "probs": p_graphsage_ensemble,
            "threshold": 0.45,
            "threshold_source": "Validation F1-max sweep on val.csv (esm150m_results/assets/evaluation/final_test_metrics.json)",
            "calibrator": "models/graph_calibrator.json on graph branch",
            "artifact": "models/ensemble_model.pkl",
            "input_dim": 7,
        },
        "ESM + GAT + XGBoost": {
            "probs": p_gat_ensemble,
            "threshold": 0.50,
            "threshold_source": "OOF stacking meta-learner evaluation threshold (ESMGAT/training/train_gat_ensemble.py, ESMGAT/results/gat_oof_metrics.json)",
            "calibrator": "ESMGAT/weights/gat_calibrator.json on graph branch",
            "artifact": "ESMGAT/weights/gat_ensemble_model.pkl",
            "input_dim": 7,
        },
    }

    metrics_results = {}
    comparison_rows = []

    for model_name, cfg in model_configs.items():
        m = compute_metrics(test_labels, cfg["probs"], cfg["threshold"])
        # Also compute at default threshold 0.50 for transparent comparison
        m_default = compute_metrics(test_labels, cfg["probs"], 0.50)

        metrics_results[model_name] = {
            "validation_derived_metrics": m,
            "default_0_5_metrics": m_default,
            "threshold": cfg["threshold"],
            "threshold_source": cfg["threshold_source"],
            "calibrator_used": cfg["calibrator"],
            "model_artifact": cfg["artifact"],
            "input_feature_dimensions": cfg["input_dim"],
        }

        comparison_rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "ROC_AUC": f"{m['roc_auc']:.4f}",
            "PR_AUC": f"{m['pr_auc']:.4f}",
            "Threshold": f"{cfg['threshold']:.2f}",
        })

    # Print Comparison Table
    comp_df = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 90)
    print("  FINAL HELD-OUT TEST EVALUATION METRICS (Validation-Derived Thresholds)")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90)

    # Also display default 0.50 table
    comp_df_def = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{res['default_0_5_metrics']['accuracy'] * 100:.2f}%",
            "Precision": f"{res['default_0_5_metrics']['precision']:.4f}",
            "Recall": f"{res['default_0_5_metrics']['recall']:.4f}",
            "F1": f"{res['default_0_5_metrics']['f1']:.4f}",
            "ROC_AUC": f"{res['default_0_5_metrics']['roc_auc']:.4f}",
            "PR_AUC": f"{res['default_0_5_metrics']['pr_auc']:.4f}",
            "Threshold": "0.50",
        }
        for name, res in metrics_results.items()
    ])
    print("\n" + "=" * 90)
    print("  SUPPLEMENTARY TEST EVALUATION METRICS (Standard Threshold = 0.50)")
    print("=" * 90)
    print(comp_df_def.to_string(index=False))
    print("=" * 90)

    # =========================================================================
    # 6. SAVE PREDICTIONS & METRICS FILES
    # =========================================================================
    print("\n[6/9] Saving final predictions and metrics artifacts...")
    results_dir = PROJECT_ROOT / "ESMGAT" / "results"
    assert_safe_gat_write(results_dir, PROJECT_ROOT)
    os.makedirs(results_dir, exist_ok=True)

    # 6a. final_predictions.csv
    pred_graphsage = (p_graphsage >= model_configs["GraphSAGE"]["threshold"]).astype(int)
    pred_gat = (p_gat >= model_configs["GAT"]["threshold"]).astype(int)
    pred_graphsage_ensemble = (p_graphsage_ensemble >= model_configs["ESM + GraphSAGE + XGBoost"]["threshold"]).astype(int)
    pred_gat_ensemble = (p_gat_ensemble >= model_configs["ESM + GAT + XGBoost"]["threshold"]).astype(int)

    preds_df = pd.DataFrame({
        "protein1": test_df["protein1"],
        "protein2": test_df["protein2"],
        "label": test_labels,
        "p_seq": p_seq,
        "p_graphsage": p_graphsage,
        "p_gat": p_gat,
        "p_graphsage_ensemble": p_graphsage_ensemble,
        "p_gat_ensemble": p_gat_ensemble,
        "pred_graphsage": pred_graphsage,
        "pred_gat": pred_gat,
        "pred_graphsage_ensemble": pred_graphsage_ensemble,
        "pred_gat_ensemble": pred_gat_ensemble,
    })

    pred_csv_path = results_dir / "final_predictions.csv"
    assert_safe_gat_write(pred_csv_path, PROJECT_ROOT)
    preds_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved final predictions to: {pred_csv_path} ({len(preds_df)} rows)")

    # 6b. final_comparison.csv
    comp_csv_path = results_dir / "final_comparison.csv"
    assert_safe_gat_write(comp_csv_path, PROJECT_ROOT)
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"  Saved comparison table to:  {comp_csv_path}")

    # 6c. final_test_metrics.json
    metrics_json_path = results_dir / "final_test_metrics.json"
    assert_safe_gat_write(metrics_json_path, PROJECT_ROOT)
    metrics_payload = {
        "test_dataset": "data/processed/test.csv",
        "test_sample_count": n_test,
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_revision_hash(),
        "models": metrics_results,
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Saved final metrics JSON to: {metrics_json_path}")

    # =========================================================================
    # 7. GAT SHAP ANALYSIS
    # =========================================================================
    print("\n[7/9] Performing SHAP analysis on GAT XGBoost ensemble...")
    shap_dir = results_dir / "shap"
    assert_safe_gat_write(shap_dir, PROJECT_ROOT)
    os.makedirs(shap_dir, exist_ok=True)

    # Use TreeExplainer on frozen XGBoost meta-learner
    explainer = shap.TreeExplainer(gat_ensemble_model)
    X_shap_df = pd.DataFrame(X_gat_meta, columns=GAT_META_FEATURE_NAMES)

    print(f"  Computing SHAP values for {len(X_shap_df)} test samples across {N_GAT_META_FEATURES} meta-features...")
    shap_values = explainer.shap_values(X_shap_df)

    # Mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_rank_order = np.argsort(-mean_abs_shap)  # descending

    shap_rows = []
    for rank, idx in enumerate(shap_rank_order, start=1):
        shap_rows.append({
            "feature": GAT_META_FEATURE_NAMES[idx],
            "mean_abs_shap": float(mean_abs_shap[idx]),
            "rank": rank,
        })
    shap_imp_df = pd.DataFrame(shap_rows)

    shap_imp_path = results_dir / "gat_shap_importance.csv"
    assert_safe_gat_write(shap_imp_path, PROJECT_ROOT)
    shap_imp_df.to_csv(shap_imp_path, index=False)
    print(f"  Saved SHAP importance table to: {shap_imp_path}")
    print("\n  GAT Ensemble SHAP Global Feature Importance:")
    for _, row in shap_imp_df.iterrows():
        print(f"    Rank {int(row['rank'])}: {row['feature']:12s} | Mean |SHAP| = {row['mean_abs_shap']:.6f}")

    # Save summary plot
    summary_plot_path = results_dir / "gat_shap_summary.png"
    summary_plot_path_shapdir = shap_dir / "gat_shap_summary.png"
    assert_safe_gat_write(summary_plot_path, PROJECT_ROOT)
    assert_safe_gat_write(summary_plot_path_shapdir, PROJECT_ROOT)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap_df, show=False)
    plt.title("GAT Ensemble (XGBoost) — SHAP Summary Plot", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(summary_plot_path_shapdir, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary plot to:    {summary_plot_path}")

    # Save SHAP data arrays
    shap_data_path = shap_dir / "gat_shap_values.npz"
    assert_safe_gat_write(shap_data_path, PROJECT_ROOT)
    np.savez_compressed(
        shap_data_path,
        shap_values=shap_values,
        base_values=explainer.expected_value,
        feature_names=GAT_META_FEATURE_NAMES,
        mean_abs_shap=mean_abs_shap,
    )
    print(f"  Saved SHAP data array to:      {shap_data_path}")

    # =========================================================================
    # 8. POST-EVALUATION ARTIFACT INTEGRITY VERIFICATION
    # =========================================================================
    print("\n[8/9] Verifying frozen GraphSAGE artifact integrity post-evaluation...")
    for path, expected_hash in frozen_files.items():
        verify_sha256(path, expected_hash)
        print(f"  [INTACT] {path.name}: SHA256 verified identical.")

    print("\n[9/9] Step 5 evaluation and SHAP analysis completed successfully.")


if __name__ == "__main__":
    main()
