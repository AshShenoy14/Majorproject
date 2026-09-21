import pytest
import numpy as np
import torch
import os
from pathlib import Path
from src.models.ensemble_model import PPIEnsemble
from src.analysis.explainability import PPIExplainer
from src.utils.paths import PROJECT_ROOT

def test_ensemble_feature_parity_seven_features():
    """PPIEnsemble._build_features constructs exactly the 7 meta-features (no bio_score)."""
    features = PPIEnsemble._build_features(np.array([0.8]), np.array([0.6]))
    assert features.shape == (1, 7), f"Expected feature shape (1, 7), got {features.shape}"
    # Order: [p_seq, p_graph, conf_seq, conf_graph, diff, max_conf, consensus]
    assert np.allclose(features[0], [0.8, 0.6, 0.3, 0.1, 0.2, 0.3, 0.48])


def test_ensemble_predict_requires_trained_model():
    """Verify that calling predict on an un-trained PPIEnsemble raises RuntimeError."""
    ensemble = PPIEnsemble()  # No checkpoint loaded

    with pytest.raises(RuntimeError, match="Meta-learner .* is not loaded or trained"):
        ensemble.predict(np.array([0.8]), np.array([0.6]), method="stacking")


def test_ensemble_rejects_stale_eight_feature_model(tmp_path):
    """A pickled meta-learner trained on 8 features must fail closed instead of silently mis-predicting."""
    import joblib
    import xgboost as xgb
    rng = np.random.RandomState(0)
    X, y = rng.rand(200, 8), rng.randint(0, 2, 200)
    path = tmp_path / "stale.pkl"
    joblib.dump(xgb.XGBClassifier(n_estimators=3).fit(X, y), path)
    with pytest.raises(RuntimeError, match="stale"):
        PPIEnsemble(str(path))


def test_graph_calibrator_applied_inside_predict(tmp_path):
    import joblib
    import xgboost as xgb
    from src.utils.calibration import PlattScaler
    rng = np.random.RandomState(0)
    X = PPIEnsemble._build_features(rng.rand(300), rng.rand(300))
    y = rng.randint(0, 2, 300)
    path = tmp_path / "ensemble_model.pkl"
    joblib.dump(xgb.XGBClassifier(n_estimators=3).fit(X, y), path)
    PlattScaler(a=2.0, b=-1.0).save(tmp_path / "graph_calibrator.json")
    ens = PPIEnsemble(str(path))
    raw = np.array([0.2, 0.9])
    cal = ens.calibrate_graph(raw)
    expected = 1 / (1 + np.exp(-(2.0 * np.log(raw / (1 - raw)) - 1.0)))
    assert np.allclose(cal, expected)
    seq = np.array([0.5, 0.5])
    assert np.allclose(ens.predict(seq, raw), ens.meta_model.predict_proba(PPIEnsemble._build_features(seq, cal))[:, 1])


def test_ensemble_soft_voting_disabled():
    """Verify that soft_voting fallback method is disabled."""
    ensemble = PPIEnsemble()
    
    with pytest.raises(ValueError, match="soft_voting fallback is disabled"):
        ensemble.predict(np.array([0.8]), np.array([0.6]), method="soft_voting")

def test_explainer_seven_features(tmp_path):
    """PPIExplainer returns one SHAP value per meta-feature (7)."""
    import joblib
    import xgboost as xgb
    rng = np.random.RandomState(0)
    X = PPIEnsemble._build_features(rng.rand(300), rng.rand(300))
    path = tmp_path / "m.pkl"
    joblib.dump(xgb.XGBClassifier(n_estimators=5).fit(X, rng.randint(0, 2, 300)), path)
    shap_vals = PPIExplainer(str(path)).explain_prediction(0.8, 0.6, 0.3, 0.1, 0.2, 0.3)
    assert shap_vals.shape[-1] == 7, f"Expected 7 SHAP values, got {shap_vals.shape[-1]}"


def test_checkpoints_exist():
    """Verify critical model checkpoints exist on disk."""
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    
    assert seq_path.exists(), f"Missing sequence model checkpoint at {seq_path}"
    assert graph_path.exists(), f"Missing graph model checkpoint at {graph_path}"
    assert ensemble_path.exists(), f"Missing ensemble model checkpoint at {ensemble_path}"
