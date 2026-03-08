from pydantic import BaseModel
from typing import List, Optional, Dict

class ProteinPair(BaseModel):
    protein1_seq: Optional[str] = None
    protein2_seq: Optional[str] = None
    protein1_id: Optional[str] = None
    protein2_id: Optional[str] = None

class PredictionResponse(BaseModel):
    interaction_probability: float
    esm_probability: float
    gat_probability: float
    confidence_score: float
    explanation: Dict[str, float] # SHAP values or similar
    
class NetworkRequest(BaseModel):
    threshold: float = 0.5
    
class NetworkResponse(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]

class BatchPredictionRequest(BaseModel):
    pairs: List[ProteinPair]
