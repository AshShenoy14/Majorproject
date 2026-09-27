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
