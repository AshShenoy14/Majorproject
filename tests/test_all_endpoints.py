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
    # the bootstrap must describe the current model: its mean sits on the final test accuracy
    final = client.get("/evaluation/final").json()
    ens_acc = next(v for k, v in final["models"].items() if "Ensemble" in k)["accuracy"]
    assert data["bootstrap_ci"]["accuracy"]["mean"] == pytest.approx(ens_acc, abs=2e-3)
    assert data["cold_start"]["cold_start_novel_protein_via_knn"]["accuracy"] == pytest.approx(0.833, abs=1e-2)
    assert data["shs27k"]["overall"]["accuracy"] == pytest.approx(0.698, abs=1e-2)
    assert data["huri"]["overall"]["roc_auc"] == pytest.approx(0.585, abs=1e-2)

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

def test_endpoint_therapeutic_targets_network_choice(client):
    known = client.get("/analysis/therapeutic-targets?limit=5&network=known").json()
    assert known and all(r["network"] == "known" and r["predicted_interactions"] == 0 for r in known)

    stats = client.get("/analysis/stats").json()
    if not stats.get("predicted_network_available"):
        pytest.skip("predicted_network.csv not generated (run scripts/build_predicted_network.py)")
    predicted = client.get("/analysis/therapeutic-targets?limit=20&network=predicted").json()
    assert all(r["network"] == "predicted" for r in predicted)
    assert any(r["predicted_interactions"] > 0 for r in predicted)
    assert stats["network"] == "predicted" and stats["num_edges"] > client.get("/analysis/stats?network=known").json()["num_edges"]


def test_slow_request_does_not_block_server(client):
    """Routes are sync (thread pool), so a long ESM computation must not freeze unrelated endpoints."""
    import random
    import threading
    import time

    rng = random.Random(0)
    seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(1000))  # novel sequence: forces an ESM pass
    worker = threading.Thread(target=lambda: client.post("/predict", json={
        "protein1_id": "SLOW_TEST_PROTEIN", "protein2_id": "ENSP00000226218", "protein1_seq": seq}))
    worker.start()
    time.sleep(0.5)
    t0 = time.time()
    res = client.get("/analysis/stats?network=known")
    elapsed = time.time() - t0
    still_running = worker.is_alive()
    worker.join()
    assert res.status_code == 200
    if not still_running:
        pytest.skip("prediction finished before the probe; machine too fast for this check")
    assert elapsed < 3.0, f"/analysis/stats took {elapsed:.1f}s while /predict was running"

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


def test_mutate_residue_mismatch_is_reported_not_500(client):
    seq = client.post("/predict", json={"protein1_id": "ENSP00000222573", "protein2_id": "ENSP00000226218"}).json()["protein1_seq"]
    wrong = "W" if seq[4] != "W" else "A"
    res = client.post("/analysis/mutate", json={
        "protein1_id": "ENSP00000222573", "protein2_id": "ENSP00000226218",
        "mutations": [{"protein": 1, "pos": 5, "orig": wrong, "mut": "A"},
                      {"protein": 1, "pos": 99999, "orig": "A", "mut": "G"}]})
    assert res.status_code == 200
    mismatch, out_of_range = res.json()["mutation_results"]
    assert "mismatch" in mismatch["error"] and mismatch["impact_delta"] is None
    assert "outside" in out_of_range["error"]


def test_optimize_unknown_protein_is_404(client):
    res = client.post("/analysis/optimize", json={"protein1_id": "NOT_A_PROTEIN", "protein2_id": "ENSP00000226218"})
    assert res.status_code == 404


def test_hotspot_job_runs_with_progress(client):
    """Long analyses run as background jobs; short custom sequences keep this test fast."""
    import time
    seq_a, seq_b = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIE", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALE"
    start = client.post("/analysis/jobs/hotspots", json={"protein1_id": "JOB_A", "protein2_id": "JOB_B",
                                                         "protein1_seq": seq_a, "protein2_seq": seq_b})
    assert start.status_code == 200
    job_id = start.json()["job_id"]
    for _ in range(240):
        job = client.get(f"/analysis/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert job["progress"]["done"] == job["progress"]["total"] > 0
    result = job["result"]
    assert len(result["protein1"]["residue_impact"]) == len(seq_a)
    assert len(result["protein2"]["residue_impact"]) == len(seq_b)
    assert client.get("/analysis/jobs/does-not-exist").status_code == 404
