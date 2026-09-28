import pytest
from fastapi.testclient import TestClient
from ESMGAT.backend.main import app

@pytest.fixture(scope="module")
def gat_client():
    with TestClient(app) as c:
        yield c

def test_gat_health_endpoint(gat_client):
    res = gat_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "Standard GAT" in data["model"]
    assert data["port"] == 8001
    assert data["role"] == "Controlled Experimental Comparison"

def test_gat_info_endpoint(gat_client):
    res = gat_client.get("/info")
    assert res.status_code == 200
    data = res.json()
    assert data["metrics"]["accuracy"] == 0.8979
    assert data["threshold"] == 0.50
    assert len(data["features"]) == 7

def test_gat_predict_endpoint(gat_client):
    payload = {
        "protein1_id": "ENSP00000269305",
        "protein2_id": "ENSP00000258149"
    }
    res = gat_client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert 0.0 <= data["interaction_probability"] <= 1.0
    assert data["role"] == "Controlled Experimental Comparison"
    assert "explanation" in data
    assert len(data["shap_explanations"]) == 7


def test_gat_predict_returns_attention_explanation(gat_client):
    """The GAT response carries per-layer attention over each protein's graph neighbours (Outcome 3)."""
    res = gat_client.post("/predict", json={"protein1_id": "ENSP00000222573", "protein2_id": "ENSP00000226218"})
    assert res.status_code == 200
    att = res.json()["attention_explanation"]
    assert att["protein1"]["graph_node"] == "ENSP00000222573"
    assert att["protein1"]["surrogate"] is False
    assert [layer["layer"] for layer in att["layers"]] == [1, 2]
    for layer in att["layers"]:
        for key in ("protein1_top", "protein2_top"):
            weights = [n["weight"] for n in layer[key]]
            assert weights == sorted(weights, reverse=True)
            assert all(0.0 <= w <= 1.0 for w in weights)
            assert all(len(n["head_weights"]) == 4 for n in layer[key])
        # a node's attention over all its neighbours (incl. itself) sums to 1, so the top-k can't exceed 1
        assert sum(n["weight"] for n in layer["protein1_top"]) <= 1.0 + 1e-5


def test_gat_attention_uses_surrogate_for_novel_protein(gat_client):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDW"
    res = gat_client.post("/predict", json={"protein1_id": "NOVEL_TEST_PROTEIN", "protein2_id": "ENSP00000226218",
                                            "protein1_seq": seq})
    assert res.status_code == 200
    att = res.json()["attention_explanation"]
    assert att["protein1"]["surrogate"] is True
    assert att["protein1"]["graph_node"].startswith("ENSP")


def test_gat_shap_values_are_computed(gat_client):
    """SHAP must come from the TreeExplainer; seven zeros means the explainer silently failed to initialise."""
    res = gat_client.post("/predict", json={"protein1_id": "ENSP00000222573", "protein2_id": "ENSP00000226218"})
    shap_values = res.json()["shap_explanations"]
    assert len(shap_values) == 7
    assert any(abs(v) > 1e-6 for v in shap_values)
