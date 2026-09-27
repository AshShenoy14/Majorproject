import hashlib
import os
from pathlib import Path
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

