import pytest
import time
from src.utils.esm_config import ESM_MODEL_NAME, ESM_EMBED_DIM

def test_quantized_esm2_inference():
    from src.data.feature_extraction import get_quantized_esm_embedding
    sequence = "MGEKSLVCSVA"
    
    start_time = time.time()
    emb = get_quantized_esm_embedding(sequence, model_name=ESM_MODEL_NAME)
    elapsed = time.time() - start_time
    
    assert emb.shape[0] == ESM_EMBED_DIM
    assert elapsed < 10.0  # Must be reasonable on CPU
