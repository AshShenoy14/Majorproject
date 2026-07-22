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
