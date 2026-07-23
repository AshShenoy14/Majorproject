import pytest
import torch
import numpy as np
from src.analysis.explainability import PPIExplainer
from src.analysis.explain_model import explain_prediction as explain_gnn
from src.utils.paths import PROJECT_ROOT, PROCESSED_DATA_DIR

def test_explainers_integration():
    # 1. SHAP Explainer test
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    assert ensemble_path.exists()
    
    explainer = PPIExplainer(str(ensemble_path))
    # mock predictions and scores
    seq_prob = 0.8
    graph_prob = 0.7
    conf_seq = 0.3
    conf_graph = 0.2
    disagreement = 0.1
    max_conf = 0.3
    bio_score = 1.0
    
    shap_vals = explainer.explain_prediction(
        seq_prob, graph_prob, conf_seq, conf_graph, disagreement, max_conf, bio_score
    )
    # SHAP values should be returned as np array or list
    assert shap_vals is not None
    assert len(shap_vals) > 0
    
    # 2. GNN Explainer test
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    assert graph_data_path.exists()
    assert mapping_path.exists()
    
    node_mapping = torch.load(mapping_path, weights_only=False)
    # Pick two valid IDs from the graph mapping
    keys = list(node_mapping.keys())
    assert len(keys) >= 2
    p1_id, p2_id = keys[0], keys[1]
    
    gnn_exp = explain_gnn(p1_id, p2_id)
    assert gnn_exp is not None
    if "error" not in gnn_exp:
        assert "top_features" in gnn_exp
        assert "top_neighbors" in gnn_exp
