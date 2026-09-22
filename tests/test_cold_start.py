import pytest
import torch
import numpy as np

def test_knn_dynamic_graph_insertion():
    # Mock node embeddings and graph
    existing_nodes = {
        "P1": torch.randn(1280),
        "P2": torch.randn(1280),
        "P3": torch.randn(1280)
    }
    novel_node_emb = torch.randn(1280)
    
    # Run the insertion function
    from app.backend.main import insert_novel_node_knn
    neighbors = insert_novel_node_knn(novel_node_emb, existing_nodes, k=2)
    
    assert len(neighbors) == 2
    assert all(n in existing_nodes for n in neighbors)


def test_cold_start_eval_artifact_exists_and_is_sane():
    """scripts/cold_start_eval.py scores the production cold-start path (insert_novel_node_knn +
    GraphSAGE decode) against real test.csv labels after physically removing proteins from the
    trained graph. This checks the committed result is present and internally consistent -- it does
    not re-run the (slow, CPU-only) evaluation itself."""
    import json
    from src.utils.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "assets" / "evaluation" / "cold_start_eval.json"
    if not path.exists():
        pytest.skip("cold_start_eval.json not generated (run scripts/cold_start_eval.py)")
    data = json.loads(path.read_text())
    cold = data["cold_start_novel_protein_via_knn"]
    warm = data["warm_baseline_same_pairs_full_graph"]
    assert cold["n"] == warm["n"] > 0
    # cold-start must beat majority-class guessing by a wide margin, and warm (fully transductive,
    # same pairs) must outperform the reconstructed cold-start path -- both are expected properties,
    # not just "any number present".
    assert cold["accuracy"] > 0.65
    assert warm["accuracy"] >= cold["accuracy"]
