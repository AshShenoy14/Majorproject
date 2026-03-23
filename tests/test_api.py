import pytest
from fastapi.testclient import TestClient
from app.backend.main import app

client = TestClient(app)

def test_read_root_docs():
    """Verify that the documentation endpoint is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_network_endpoint():
    """Test the network subgraph retrieval."""
    response = client.get("/network?limit=5")
    # Even if train.csv is missing, it should return 200 with empty lists or handle it
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

def test_predict_validation_error():
    """Test that invalid protein IDs return 400."""
    response = client.post("/predict", json={"protein1_id": "", "protein2_id": ""})
    assert response.status_code == 400

def test_bio_metadata_endpoint():
    """Test biological metadata retrieval."""
    # Using a dummy ID
    response = client.get("/bio/metadata?proteins=ENSP00000327694")
    assert response.status_code in [200, 503]
