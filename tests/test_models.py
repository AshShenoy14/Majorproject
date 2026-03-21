import torch
import pytest
import numpy as np
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble

def test_sequence_model_forward():
    input_dim = 480
    batch_size = 4
    total_dim = input_dim + 10 # Default bio_dim is 10
    model = SequencePPIModel(input_dim=input_dim)
    model.eval()
    
    emb1 = torch.randn(batch_size, total_dim)
    emb2 = torch.randn(batch_size, total_dim)
    
    with torch.no_grad():
        output = model(emb1, emb2)
    
    assert output.shape == (batch_size, 1)
    # Check if logits are within reasonable range or at least exist
    assert not torch.isnan(output).any()

def test_graph_model_forward():
    in_channels = 480
    num_nodes = 10
    num_edges = 20
    num_pairs = 5
    
    total_dim = in_channels # GNN model uses raw x as input, no bio concat inside forward
    model = GATLinkPredictor(in_channels=in_channels, hidden_channels=128)
    model.eval()
    
    x = torch.randn(num_nodes, total_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_label_index = torch.randint(0, num_nodes, (2, num_pairs))
    
    with torch.no_grad():
        output = model(x, edge_index, edge_label_index)
        
    assert output.shape == (num_pairs, 1)
    assert not torch.isnan(output).any()

def test_ensemble_soft_voting():
    ensemble = PPIEnsemble()
    preds1 = np.array([0.8, 0.2, 0.9])
    preds2 = np.array([0.7, 0.4, 0.1])
    
    result = ensemble.predict(preds1, preds2, method="soft_voting")
    expected = (preds1 + preds2) / 2.0
    
    np.testing.assert_allclose(result, expected)

def test_ensemble_feature_building():
    ensemble = PPIEnsemble()
    preds1 = np.array([0.8, 0.2])
    preds2 = np.array([0.7, 0.4])
    
    features = ensemble._build_features(preds1, preds2)
    # Expected: [p1, p2, |p1-0.5|, |p2-0.5|]
    assert features.shape == (2, 4)
    assert features[0, 2] == pytest.approx(0.3) # |0.8 - 0.5|
    assert features[1, 2] == pytest.approx(0.3) # |0.2 - 0.5|
