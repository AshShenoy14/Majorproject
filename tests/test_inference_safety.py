import pytest
import numpy as np
import torch
import os
from pathlib import Path
from src.models.ensemble_model import PPIEnsemble
from src.analysis.explainability import PPIExplainer
from src.utils.paths import PROJECT_ROOT

def test_ensemble_feature_parity_eight_features():
    """Verify that PPIEnsemble._build_features constructs exactly 8 features including consensus."""
    base_preds_1 = np.array([0.8])
    base_preds_2 = np.array([0.6])
    bio_features = np.array([[0.9]])
    
    features = PPIEnsemble._build_features(base_preds_1, base_preds_2, bio_features)
    
    assert features.shape == (1, 8), f"Expected feature shape (1, 8), got {features.shape}"
    
    # Feature order: [seq, graph, conf_1, conf_2, disagreement, max_conf, consensus, bio]
    # Check consensus calculation (seq * graph = 0.8 * 0.6 = 0.48)
    expected_consensus = 0.8 * 0.6
    assert np.isclose(features[0, 6], expected_consensus), f"Expected consensus {expected_consensus}, got {features[0, 6]}"
    assert np.isclose(features[0, 7], 0.9), f"Expected bio score 0.9, got {features[0, 7]}"

def test_ensemble_predict_requires_trained_model():
    """Verify that calling predict on an un-trained PPIEnsemble raises RuntimeError."""
    ensemble = PPIEnsemble() # No checkpoint loaded
    
    with pytest.raises(RuntimeError, match="Meta-learner .* is not loaded or trained"):
        ensemble.predict(np.array([0.8]), np.array([0.6]), bio_features=np.array([[0.9]]), method="stacking")

def test_ensemble_soft_voting_disabled():
    """Verify that soft_voting fallback method is disabled."""
    ensemble = PPIEnsemble()
    
    with pytest.raises(ValueError, match="soft_voting fallback is disabled"):
        ensemble.predict(np.array([0.8]), np.array([0.6]), method="soft_voting")

def test_explainer_eight_features():
    """Verify PPIExplainer feature alignment."""
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    if ensemble_path.exists():
        explainer = PPIExplainer(str(ensemble_path))
        shap_vals = explainer.explain_prediction(0.8, 0.6, 0.3, 0.1, 0.2, 0.3, 0.9)
        assert shap_vals.shape[-1] == 8, f"Expected 8 SHAP values, got {shap_vals.shape[-1]}"

def test_checkpoints_exist():
    """Verify critical model checkpoints exist on disk."""
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    
    assert seq_path.exists(), f"Missing sequence model checkpoint at {seq_path}"
    assert graph_path.exists(), f"Missing graph model checkpoint at {graph_path}"
    assert ensemble_path.exists(), f"Missing ensemble model checkpoint at {ensemble_path}"
