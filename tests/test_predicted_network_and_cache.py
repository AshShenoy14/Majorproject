"""
Tests for the predicted-network analysis (Objective 3), GAT attention extraction (Outcome 3) and the
sequence-keyed ESM embedding cache.
"""
import json

import pandas as pd
import pytest
import torch

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from ESMGAT.models.gat_model import GATLinkPredictor

PREDICTED_CSV = PROCESSED_DATA_DIR / "predicted_network.csv"
SUMMARY_JSON = PROJECT_ROOT / "assets" / "evaluation" / "predicted_network_summary.json"


def test_encode_with_attention_matches_encode():
    """encode_with_attention() must not change the embeddings, and attention must be a distribution per node."""
    torch.manual_seed(0)
    model = GATLinkPredictor(in_channels=16, hidden_channels=32, heads=4).eval()
    x = torch.randn(20, 16)
    edge_index = torch.randint(0, 20, (2, 60))
    with torch.no_grad():
        z_plain = model.encode(x, edge_index)
        z_att, attention = model.encode_with_attention(x, edge_index)
    assert torch.allclose(z_plain, z_att)
    assert len(attention) == 2
    for ei, alpha in attention:
        assert alpha.shape == (ei.shape[1], 4)
        per_node = torch.zeros(20, 4).index_add_(0, ei[1], alpha)
        assert torch.allclose(per_node, torch.ones(20, 4), atol=1e-5)


@pytest.fixture(scope="module")
def predicted_network():
    if not PREDICTED_CSV.exists():
        pytest.skip("predicted_network.csv not generated (run scripts/build_predicted_network.py)")
    return pd.read_csv(PREDICTED_CSV)


def test_predicted_network_composition(predicted_network):
    summary = json.loads(SUMMARY_JSON.read_text())
    known = predicted_network[predicted_network["source"] == "known"]
    predicted = predicted_network[predicted_network["source"] == "predicted"]

    # known edges are exactly the train positives; predicted edges come only from held-out splits
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    assert len(known) == int((train["label"] == 1).sum()) == summary["known_edges"]
    assert set(predicted["split"]) <= {"val", "test"}
    assert (predicted["probability"] >= summary["threshold"]).all()
    assert len(predicted) == summary["predicted_edges"]

    # splits are pair-disjoint, so no predicted edge duplicates a training edge
    key = lambda df: set(map(frozenset, zip(df["protein1"], df["protein2"])))
    assert not key(known) & key(predicted)

    # the scoring reproduced the committed test-set predictions
    assert summary["reproduction_check"]["max_abs_probability_diff"] < 1e-4


@pytest.fixture(scope="module")
def extractor():
    from src.data.feature_extraction import ESMFeatureExtractor
    ex = ESMFeatureExtractor(device="cpu")
    if not ex.cache:
        pytest.skip("embedding cache not seeded (embeddings.pt / sequences_cache.json missing)")
    return ex


def test_embedding_cache_is_keyed_by_sequence(extractor):
    """A mutated sequence under a known protein ID must get its own embedding, not the cached wild type."""
    seqs = json.loads((PROCESSED_DATA_DIR / "sequences_cache.json").read_text())
    pid = "ENSP00000222573"
    wild_type = seqs[pid]
    stored = extractor.get_embeddings({pid: wild_type})[pid]
    assert stored is extractor.cache[wild_type]  # served from the cache, no ESM pass

    pos = 4
    mutant = wild_type[:pos] + ("W" if wild_type[pos] != "W" else "A") + wild_type[pos + 1:]
    mutated = extractor.get_embeddings({pid: mutant})[pid]
    assert not torch.equal(mutated, stored)
    assert extractor.cache[mutant] is mutated
