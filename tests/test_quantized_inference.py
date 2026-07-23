import pytest
import time

def test_quantized_esm2_inference():
    from src.data.feature_extraction import get_quantized_esm_embedding
    sequence = "MGEKSLVCSVA"
    
    start_time = time.time()
    emb = get_quantized_esm_embedding(sequence, model_name="facebook/esm2_t12_35M_UR50D")
    elapsed = time.time() - start_time
    
    # t12_35M dimension is 480
    assert emb.shape[0] == 480
    assert elapsed < 10.0  # Must be reasonable on CPU
