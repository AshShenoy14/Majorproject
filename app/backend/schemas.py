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
    protein1_uniprot_id: Optional[str] = None
    protein2_uniprot_id: Optional[str] = None
    
class NetworkRequest(BaseModel):
    threshold: float = 0.5
    
class NetworkResponse(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]

class BatchPredictionRequest(BaseModel):
    pairs: List[ProteinPair]

class MutationItem(BaseModel):
    protein: int # 1 or 2
    pos: int
    orig: str
    mut: str

class MutationRequest(BaseModel):
    protein1_id: str
    protein1_seq: str
    protein2_id: str
    protein2_seq: str
    mutations: List[MutationItem]

class MutationResult(BaseModel):
    protein: int
    pos: int
    orig: str
    mut: str
    base_score: float
    mutated_score: float
    impact_delta: float
    interpretation: str
    error: Optional[str] = None

class MutationAnalysisResponse(BaseModel):
    protein1: str
    protein2: str
    mutation_results: List[MutationResult]

class BioMetaResponse(BaseModel):
    protein_id: str
    uniprot_id: str
    localization: str
    pathways: str

class FeasibilityResponse(BaseModel):
    compatible: bool
    intersection: List[str]
    p1_locs: List[str]
    p2_locs: List[str]
    reason: str
