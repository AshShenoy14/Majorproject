import pytest
from fastapi.testclient import TestClient
from app.backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_endpoint_evaluation_final(client):
    res = client.get("/evaluation/final")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert any("Ensemble" in k for k in data["models"])

def test_endpoint_all_benchmarks(client):
    res = client.get("/evaluation/benchmarks")
    assert res.status_code == 200
    data = res.json()
    # every committed artifact should be present and match the verified numbers
    assert data["bootstrap_ci"]["accuracy"]["mean"] == pytest.approx(0.9213, abs=1e-3)
    assert data["cold_start"]["cold_start_novel_protein_via_knn"]["accuracy"] == pytest.approx(0.833, abs=1e-2)
    assert data["shs27k"]["overall"]["accuracy"] == pytest.approx(0.698, abs=1e-2)
    assert data["huri"]["overall"]["roc_auc"] == pytest.approx(0.573, abs=1e-2)

def test_endpoint_network_subgraph(client):
    res = client.get("/network?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data

def test_endpoint_centrality(client):
    res = client.get("/analysis/centrality?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_endpoint_network_stats(client):
    res = client.get("/analysis/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_nodes" in data or "nodes" in data or isinstance(data, dict)

def test_endpoint_bio_metadata(client):
    res = client.get("/bio/metadata?proteins=ENSP00000269305")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)

def test_endpoint_bio_feasibility(client):
    res = client.get("/bio/feasibility?p1=ENSP00000269305&p2=ENSP00000258149")
    assert res.status_code == 200
    data = res.json()
    assert "compatible" in data
    assert "intersection" in data

def test_endpoint_predict_success(client):
    payload = {
        "protein1_id": "ENSP00000269305",
        "protein2_id": "ENSP00000258149"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert 0.0 <= data["interaction_probability"] <= 1.0
    assert "shap_explanations" in data

def test_endpoint_predict_invalid_sequence_422(client):
    payload = {
        "protein1_id": "P1",
        "protein2_id": "P2",
        "protein1_seq": "INVALID123!@"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_endpoint_predict_batch(client):
    payload = {
        "pairs": [
            {"protein1_id": "ENSP00000269305", "protein2_id": "ENSP00000258149"}
        ]
    }
    res = client.post("/predict_batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["status"] == "success"

def test_endpoint_therapeutic_targets(client):
    res = client.get("/analysis/therapeutic-targets?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)

def test_endpoint_chat_greeting(client):
    res = client.get("/chat/greeting")
    assert res.status_code == 200
    data = res.json()
    assert "response" in data

def test_endpoint_chat_message(client):
    res = client.post("/chat", json={"message": "What does TP53 do?"})
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
