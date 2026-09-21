import pytest
from pydantic import ValidationError
from app.backend.schemas import ProteinPair, PredictionResponse, FeasibilityResponse
from src.analysis.biological_managers import BiologicalManager
from src.analysis.explain_model import get_topological_neighbors

def test_protein_pair_sequence_validation_valid():
    """Test that valid canonical amino acid sequences pass validation and are cleaned."""
    pair = ProteinPair(
        protein1_id="P1",
        protein2_id="P2",
        protein1_seq="  m a h v l  ",
        protein2_seq="ACDEFGHIKLMNPQRSTVWY"
    )
    assert pair.protein1_seq == "MAHVL"
    assert pair.protein2_seq == "ACDEFGHIKLMNPQRSTVWY"

def test_protein_pair_sequence_validation_invalid_chars():
    """Test that sequences with invalid characters (e.g., '123!!@@') fail with ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        ProteinPair(
            protein1_id="P1",
            protein2_id="P2",
            protein1_seq="123!!@@"
        )
    assert "Invalid amino acid character" in str(excinfo.value)

def test_protein_pair_sequence_validation_too_short():
    """Test that sequences shorter than 5 residues fail validation."""
    with pytest.raises(ValidationError) as excinfo:
        ProteinPair(
            protein1_id="P1",
            protein2_id="P2",
            protein1_seq="MA"
        )
    assert "at least 5 amino acids" in str(excinfo.value)

def test_feasibility_response_schema_compatibility():
    """Verify that BiologicalManager check_biological_compatibility output validates cleanly against FeasibilityResponse."""
    manager = BiologicalManager()
    result = manager.check_biological_compatibility("UNKNOWN_1", "UNKNOWN_2", fetch_missing=False, persist=False)
    assert "intersection" in result
    
    validated = FeasibilityResponse(**result)
    assert validated.compatible is True
    assert validated.intersection == []

def test_fast_topological_neighbors():
    """Verify get_topological_neighbors returns formatted neighbors within milliseconds."""
    res = get_topological_neighbors("ENSP00000327694", "ENSP00000373627")
    assert "top_neighbors" in res
    assert "top_features" in res
    assert isinstance(res["top_neighbors"], list)
    if res["top_neighbors"]:
        first = res["top_neighbors"][0]
        assert "id" in first
        assert "importance" in first

def test_prediction_response_error_format():
    """Verify PredictionResponse supports clean error representation without numeric fakes."""
    err_resp = PredictionResponse(
        protein1_id="P1",
        protein2_id="P2",
        status="error",
        error="Could not resolve sequence"
    )
    assert err_resp.status == "error"
    assert err_resp.interaction_probability is None
    assert err_resp.error == "Could not resolve sequence"
