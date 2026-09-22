"""
Sanity checks for the committed external-benchmark artifacts (scripts/external_benchmark_shs27k.py
and scripts/external_benchmark_huri.py). Does not re-run either benchmark (both are slow: SHS27k
extracts ESM-2 embeddings for ~300 sequences on CPU, HuRI additionally makes real network calls to
the Ensembl REST API) -- it only checks that a committed result exists and is internally consistent.
"""
import json
import pytest

from src.utils.paths import PROJECT_ROOT


def _load(name):
    path = PROJECT_ROOT / "assets" / "evaluation" / name
    if not path.exists():
        pytest.skip(f"{name} not generated (run the corresponding scripts/external_benchmark_*.py)")
    return json.loads(path.read_text())


def test_shs27k_benchmark_artifact_is_sane():
    data = _load("external_benchmark_shs27k.json")
    overall = data["overall"]
    assert overall["n"] == data["n_total_pairs"] > 0
    assert 0.0 <= overall["accuracy"] <= 1.0
    assert 0.0 <= overall["roc_auc"] <= 1.0
    # SHS27k is same-source (STRING) as training data, so it should still clearly beat a coin flip,
    # even though it scores well below the in-domain 92% test-set result.
    assert overall["accuracy"] > 0.55
    assert overall["roc_auc"] > 0.6


def test_huri_benchmark_artifact_is_sane():
    data = _load("external_benchmark_huri.json")
    overall = data["overall"]
    assert overall["n"] > 0
    assert 0.0 <= overall["accuracy"] <= 1.0
    assert 0.0 <= overall["roc_auc"] <= 1.0
    assert overall["roc_auc"] > 0.5, "should beat random guessing even on an independent-source dataset"
