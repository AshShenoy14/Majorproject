import hashlib
import os
from pathlib import Path
import numpy as np
import pytest
import torch

from ESMGAT.models.gat_model import GATLinkPredictor
from src.utils.paths import PROJECT_ROOT, MODELS_DIR, CHECKPOINT_DIR


def test_gat_import_and_instantiation():
    """Test A & B: GAT model imports and instantiates with required dimensions on CPU."""
    model = GATLinkPredictor(in_channels=483, hidden_channels=256, heads=4)
    assert isinstance(model, torch.nn.Module)
    assert model.in_channels == 483
    assert model.hidden_channels == 256
    assert model.heads == 4


def test_gat_attention_heads_and_hidden_dims():
    """Test C & D: Layer 1 has 4 heads and intermediate representations have 256 dims."""
    model = GATLinkPredictor(in_channels=483, hidden_channels=256, heads=4)
    assert model.conv1.heads == 4
    assert model.conv2.heads == 4
    # conv1: in_channels (483) -> heads (4) * out_channels (64) = 256
    assert model.conv1.out_channels == 64
    # conv2: hidden_channels (256) -> out_channels (256) with concat=False
    assert model.conv2.out_channels == 256
    assert model.conv2.concat is False


def test_gat_decoder_input_dimension():
    """Test J: Decoder input dimension is strictly 1025 (256*4 + 1)."""
    model = GATLinkPredictor(in_channels=483, hidden_channels=256, heads=4)
    expected_dim = (256 * 4) + 1  # [u, v, |u - v|, u * v, bilinear]
    assert model.classifier[0].in_features == expected_dim
    assert expected_dim == 1025


def test_gat_encode_and_forward_on_synthetic_graph():
    """Test E, F, G, H, I, K: Forward and encode pass on small synthetic graph on CPU."""
    device = torch.device("cpu")
    model = GATLinkPredictor(in_channels=483, hidden_channels=256, heads=4).to(device)
    model.eval()

    # Create a small synthetic graph with 12 nodes and random connections
    torch.manual_seed(42)
    num_nodes = 12
    x = torch.randn(num_nodes, 483, device=device)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 6, 8, 10, 1],
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        z = model.encode(x, edge_index)

    # Check embedding shape and finiteness
    assert z.shape == (num_nodes, 256)
    assert torch.isfinite(z).all(), "Node embeddings contain non-finite values"
    assert not torch.isnan(z).any(), "Node embeddings contain NaN values"

    # Test decoder on multiple protein pairs
    src_nodes = torch.tensor([0, 1, 3, 5, 7], dtype=torch.long, device=device)
    dst_nodes = torch.tensor([2, 4, 6, 8, 10], dtype=torch.long, device=device)
    num_pairs = len(src_nodes)

    with torch.no_grad():
        pair_logits = model.decode(z, src_nodes, dst_nodes)

    # Verify output shape [num_pairs, 1]
    assert pair_logits.shape == (num_pairs, 1)
    assert torch.isfinite(pair_logits).all(), "Decoder outputs contain non-finite values"
    assert not torch.isnan(pair_logits).any(), "Decoder outputs contain NaN values"

    # Test end-to-end forward method
    edge_label_index = torch.stack([src_nodes, dst_nodes])
    with torch.no_grad():
        direct_out = model(x, edge_index, edge_label_index)

    assert direct_out.shape == (num_pairs, 1)
    assert torch.allclose(pair_logits, direct_out, atol=1e-5)


def test_graphsage_files_unmodified():
    """Test L: Existing GraphSAGE source files and production models have not been modified."""
    # 1. Source files exist and maintain their expected classes
    sage_file = PROJECT_ROOT / "src" / "models" / "graph_model.py"
    assert sage_file.exists()
    content = sage_file.read_text(encoding="utf-8")
    assert "class SAGELinkPredictor(nn.Module):" in content
    assert "conv1 = SAGEConv" in content

    # 2. Check training and base trainer files
    assert (PROJECT_ROOT / "src" / "training" / "train_graph_model.py").exists()
    assert (PROJECT_ROOT / "src" / "training" / "base_trainers.py").exists()
    assert (PROJECT_ROOT / "src" / "training" / "train_ensemble.py").exists()

    # 3. Verify stable model weights have not been overwritten
    sage_model = MODELS_DIR / "graph_model_best.pth"
    assert sage_model.exists()
    assert sage_model.stat().st_size == 4764435
    h_sage = hashlib.sha256(sage_model.read_bytes()).hexdigest()[:16]
    assert h_sage == "16d3b16ea464f5e1"

    # 4. Verify stable ensemble model has not been overwritten
    ens_model = MODELS_DIR / "ensemble_model.pkl"
    assert ens_model.exists()
    assert ens_model.stat().st_size == 3091241
    h_ens = hashlib.sha256(ens_model.read_bytes()).hexdigest()[:16]
    assert h_ens == "7e2c9543cb16f194"


def test_gat_training_path_safety():
    """Verify that assert_safe_path strictly forbids writing to root models/ or checkpoints/."""
    from ESMGAT.training.gat_base_trainers import assert_safe_path

    # Valid ESMGAT targets must pass without error
    valid_target = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_model_best.pth"
    assert_safe_path(valid_target, PROJECT_ROOT)

    valid_ckpt = PROJECT_ROOT / "ESMGAT" / "checkpoints" / "gat" / "gat_checkpoint.pt"
    assert_safe_path(valid_ckpt, PROJECT_ROOT)

    # Attempts to target stable models/ or checkpoints/ must raise RuntimeError
    forbidden_target1 = PROJECT_ROOT / "models" / "gat_model_best.pth"
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
        assert_safe_path(forbidden_target1, PROJECT_ROOT)

    forbidden_target2 = PROJECT_ROOT / "checkpoints" / "gat_checkpoint.pt"
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
        assert_safe_path(forbidden_target2, PROJECT_ROOT)


def test_trained_gat_checkpoint_and_calibrator_load():
    """Verify that the trained GAT weights and calibrator downloaded from Colab load cleanly."""
    from src.utils.calibration import PlattScaler

    gat_weights_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_model_best.pth"
    gat_cal_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_calibrator.json"

    assert gat_weights_path.exists(), "gat_model_best.pth not found in ESMGAT/weights/"
    assert gat_cal_path.exists(), "gat_calibrator.json not found in ESMGAT/weights/"

    state_dict = torch.load(gat_weights_path, map_location="cpu", weights_only=False)
    in_channels = state_dict["input_norm.weight"].shape[0]
    model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256, heads=4)
    model.load_state_dict(state_dict)
    model.eval()

    assert model.in_channels == 643
    assert model.hidden_channels == 256

    scaler = PlattScaler.load(gat_cal_path)
    assert isinstance(scaler.a, float) and isinstance(scaler.b, float)
    assert scaler.a > 0, "Platt scaler coefficient 'a' must be positive"


def test_gat_meta_features_construction():
    """Verify that the 7 meta-features are constructed with exact mathematical correctness."""
    from ESMGAT.models.gat_ensemble import GATPPIEnsemble, GAT_META_FEATURE_NAMES, N_GAT_META_FEATURES

    assert len(GAT_META_FEATURE_NAMES) == 7
    assert N_GAT_META_FEATURES == 7

    # Synthetic test probabilities
    p_seq = np.array([0.1, 0.5, 0.9, 0.7, 0.3])
    p_gat = np.array([0.2, 0.5, 0.8, 0.4, 0.6])

    X = GATPPIEnsemble.build_meta_features(p_seq, p_gat)
    assert X.shape == (5, 7)

    # 1. p_seq
    assert np.allclose(X[:, 0], p_seq)
    # 2. p_gat
    assert np.allclose(X[:, 1], p_gat)
    # 3. abs(p_seq - 0.5)
    assert np.allclose(X[:, 2], np.abs(p_seq - 0.5))
    # 4. abs(p_gat - 0.5)
    assert np.allclose(X[:, 3], np.abs(p_gat - 0.5))
    # 5. abs(p_seq - p_gat)
    assert np.allclose(X[:, 4], np.abs(p_seq - p_gat))
    # 6. max(conf_seq, conf_gat)
    assert np.allclose(X[:, 5], np.maximum(np.abs(p_seq - 0.5), np.abs(p_gat - 0.5)))
    # 7. p_seq * p_gat
    assert np.allclose(X[:, 6], p_seq * p_gat)

    # Test error on mismatched lengths
    with pytest.raises(ValueError):
        GATPPIEnsemble.build_meta_features(p_seq[:3], p_gat)


def test_gat_safe_path_assertions():
    """Verify that assert_safe_gat_write rejects any writes outside ESMGAT/ or into frozen dirs."""
    from ESMGAT.models.gat_ensemble import assert_safe_gat_write

    # Valid ESMGAT targets
    assert_safe_gat_write(PROJECT_ROOT / "ESMGAT" / "weights" / "gat_ensemble_model.pkl")
    assert_safe_gat_write(PROJECT_ROOT / "ESMGAT" / "checkpoints" / "oof" / "gat_fold1.npz")
    assert_safe_gat_write(PROJECT_ROOT / "ESMGAT" / "results" / "gat_oof_metrics.json")

    # Forbidden paths targeting models/ or checkpoints/
    with pytest.raises(RuntimeError, match="CRITICAL PATH SAFETY VIOLATION"):
        assert_safe_gat_write(PROJECT_ROOT / "models" / "ensemble_model.pkl")

    with pytest.raises(RuntimeError, match="CRITICAL PATH SAFETY VIOLATION"):
        assert_safe_gat_write(PROJECT_ROOT / "checkpoints" / "oof" / "gat_fold1.npz")

    with pytest.raises(RuntimeError, match="CRITICAL PATH SAFETY VIOLATION"):
        assert_safe_gat_write(PROJECT_ROOT / "src" / "models" / "gat_model.py")


def test_frozen_graphsage_and_ensemble_sha256_exact():
    """Verify that all three stable GraphSAGE production files remain byte-identical."""
    expected_hashes = {
        MODELS_DIR / "graph_model_best.pth": "16d3b16ea464f5e1bd435c8c2bd3aac2cf2415a8f50fb03745f87bd4aa5d57fa",
        MODELS_DIR / "ensemble_model.pkl": "7e2c9543cb16f19429292b449128e03ace1bd88cc115c1db79fa29f1c8e98b60",
        MODELS_DIR / "graph_calibrator.json": "dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd",
    }

    for path, expected_sha in expected_hashes.items():
        assert path.exists(), f"Stable file {path} missing!"
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, (
            f"CRITICAL VIOLATION: Stable file {path.name} was modified!\n"
            f"Expected: {expected_sha}\nActual:   {actual_sha}"
        )


def test_gat_oof_files_and_coverage_if_ready():
    """Verify GAT OOF file integrity, 5-fold coverage, and non-overlapping indices."""
    oof_dir = PROJECT_ROOT / "ESMGAT" / "checkpoints" / "oof"
    fold_files = [oof_dir / f"gat_fold{i}.npz" for i in range(1, 6)]

    # If all 5 files exist, perform full verification
    if all(f.exists() for f in fold_files):
        all_ho_indices = []
        total_samples = 0

        for i, fpath in enumerate(fold_files):
            c = np.load(fpath, allow_pickle=False)
            keys = set(c.keys())
            required_keys = {"fingerprint", "architecture", "ho_idx", "seq", "gat", "gat_logit", "gat_cal", "labels"}
            assert required_keys.issubset(keys), f"Missing keys in {fpath.name}: {required_keys - keys}"

            assert str(c["architecture"]) == "GAT", f"Architecture in {fpath.name} is not GAT!"
            assert str(c["fingerprint"]).startswith("gat-"), f"Fingerprint in {fpath.name} must start with 'gat-'"

            ho_idx = c["ho_idx"]
            n_fold = len(ho_idx)
            total_samples += n_fold
            all_ho_indices.extend(ho_idx.tolist())

            # Check probabilities validity
            assert not np.isnan(c["gat"]).any(), f"NaN in raw GAT probs {fpath.name}"
            assert not np.isnan(c["gat_cal"]).any(), f"NaN in cal GAT probs {fpath.name}"
            assert not np.isnan(c["seq"]).any(), f"NaN in seq probs {fpath.name}"
            assert (c["gat_cal"] >= 0.0).all() and (c["gat_cal"] <= 1.0).all()

        # Check total count and uniqueness
        assert total_samples == 161369, f"Expected 161369 total OOF samples, got {total_samples}"
        assert len(set(all_ho_indices)) == 161369, "Duplicate pair indices found across folds!"
        assert min(all_ho_indices) == 0
        assert max(all_ho_indices) == 161368


def test_gat_ensemble_artifact_if_ready():
    """Verify GAT XGBoost meta-learner artifact loading and prediction if ready."""
    ensemble_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_ensemble_model.pkl"
    if ensemble_path.exists():
        from ESMGAT.models.gat_ensemble import GATPPIEnsemble
        ens = GATPPIEnsemble(str(ensemble_path))
        assert ens.meta_model is not None
        assert ens.meta_model.n_features_in_ == 7

        # Test predict_proba
        p_seq = np.array([0.8, 0.2])
        p_gat_raw = np.array([0.7, 0.3])
        preds = ens.predict_proba(p_seq, p_gat_raw)
        assert len(preds) == 2
        assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_step5_test_set_isolation_and_count():
    """Verify test.csv count is 20,172 and has zero leakage into train.csv."""
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    assert test_path.exists() and train_path.exists()

    import pandas as pd
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    assert len(test_df) == 20172
    assert len(train_df) == 161369

    train_pairs_sym = set()
    for p1, p2 in zip(train_df["protein1"], train_df["protein2"]):
        train_pairs_sym.add((p1, p2))
        train_pairs_sym.add((p2, p1))

    test_pairs = list(zip(test_df["protein1"], test_df["protein2"]))
    leakage = [p for p in test_pairs if p in train_pairs_sym]
    assert len(leakage) == 0, f"Detected {len(leakage)} leaking pairs in test.csv!"


def test_step5_final_predictions_file():
    """Verify final_predictions.csv row count, required columns, and probability validity."""
    pred_path = PROJECT_ROOT / "ESMGAT" / "results" / "final_predictions.csv"
    assert pred_path.exists(), f"Missing final predictions file: {pred_path}"

    import pandas as pd
    df = pd.read_csv(pred_path)
    assert len(df) == 20172

    expected_cols = [
        "protein1", "protein2", "label",
        "p_seq", "p_graphsage", "p_gat",
        "p_graphsage_ensemble", "p_gat_ensemble",
        "pred_graphsage", "pred_gat",
        "pred_graphsage_ensemble", "pred_gat_ensemble"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in final_predictions.csv"

    prob_cols = ["p_seq", "p_graphsage", "p_gat", "p_graphsage_ensemble", "p_gat_ensemble"]
    for col in prob_cols:
        assert not df[col].isna().any(), f"NaN found in column {col}"
        assert not np.isinf(df[col]).any(), f"Inf found in column {col}"
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Values outside [0, 1] in {col}"

    pred_cols = ["pred_graphsage", "pred_gat", "pred_graphsage_ensemble", "pred_gat_ensemble"]
    for col in pred_cols:
        assert set(df[col].unique()).issubset({0, 1}), f"Non-binary predictions in {col}"


def test_step5_final_metrics_and_threshold_provenance():
    """Verify final_test_metrics.json content, model count, and threshold provenance."""
    metrics_path = PROJECT_ROOT / "ESMGAT" / "results" / "final_test_metrics.json"
    comp_path = PROJECT_ROOT / "ESMGAT" / "results" / "final_comparison.csv"
    assert metrics_path.exists()
    assert comp_path.exists()

    import json
    with open(metrics_path, "r") as f:
        data = json.load(f)

    assert data["test_sample_count"] == 20172
    models = data["models"]
    expected_models = [
        "ESM-MLP", "GraphSAGE", "GAT",
        "ESM + GraphSAGE + XGBoost", "ESM + GAT + XGBoost"
    ]
    for m in expected_models:
        assert m in models, f"Model {m} missing from final_test_metrics.json"
        m_info = models[m]
        assert "threshold" in m_info
        assert "threshold_source" in m_info
        assert len(m_info["threshold_source"]) > 0
        v_metrics = m_info["validation_derived_metrics"]
        assert 0.0 <= v_metrics["accuracy"] <= 1.0
        assert 0.0 <= v_metrics["f1"] <= 1.0
        assert 0.0 <= v_metrics["roc_auc"] <= 1.0
        assert 0.0 <= v_metrics["pr_auc"] <= 1.0
        assert v_metrics["tp"] + v_metrics["tn"] + v_metrics["fp"] + v_metrics["fn"] == 20172


def test_step5_gat_shap_outputs():
    """Verify SHAP importance table, summary plot, and data arrays."""
    shap_csv = PROJECT_ROOT / "ESMGAT" / "results" / "gat_shap_importance.csv"
    shap_png = PROJECT_ROOT / "ESMGAT" / "results" / "gat_shap_summary.png"
    shap_npz = PROJECT_ROOT / "ESMGAT" / "results" / "shap" / "gat_shap_values.npz"

    assert shap_csv.exists()
    assert shap_png.exists()
    assert shap_npz.exists()

    import pandas as pd
    df = pd.read_csv(shap_csv)
    assert len(df) == 7
    assert list(df.columns) == ["feature", "mean_abs_shap", "rank"]
    assert list(df["rank"]) == [1, 2, 3, 4, 5, 6, 7]
    assert (df["mean_abs_shap"] >= 0.0).all()

    # Check npz contents
    npz = np.load(shap_npz, allow_pickle=False)
    assert "shap_values" in npz
    assert npz["shap_values"].shape == (20172, 7)
    assert len(npz["feature_names"]) == 7
