from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict

class ProteinPair(BaseModel):
    protein1_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 1", example="MAH...")
    protein2_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 2", example="MVK...")
    protein1_id: Optional[str] = Field(None, description="Identifier for protein 1 (e.g., ENSP ID)", example="ENSP00000327694")
    protein2_id: Optional[str] = Field(None, description="Identifier for protein 2 (e.g., ENSP ID)", example="ENSP00000373627")

class PredictionResponse(BaseModel):
    interaction_probability: float = Field(..., description="Final ensemble prediction probability", example=0.88)
    esm_probability: float = Field(..., description="Probability from the ESM-MLP sequence model", example=0.92)
    gat_probability: float = Field(..., description="Probability from the GAT graph model", example=0.75)
    confidence_score: float = Field(..., description="Normalized confidence score [0, 1]", example=0.76)
    explanation: Dict[str, Any] = Field(..., description="Feature importance scores (e.g., SHAP values)")
    gnn_explanation: Optional[Dict[str, Any]] = Field(None, description="Detailed GNN-specific neighbor importance")
    protein1_uniprot_id: Optional[str] = Field(None, description="Mapped UniProt ID for protein 1", example="P12345")
    protein2_uniprot_id: Optional[str] = Field(None, description="Mapped UniProt ID for protein 2", example="Q67890")
    protein1_seq: Optional[str] = Field(None, description="Sequence used for protein 1")
    protein2_seq: Optional[str] = Field(None, description="Sequence used for protein 2")
    
class NetworkRequest(BaseModel):
    threshold: float = Field(0.5, description="Interaction probability threshold for network inclusion", ge=0, le=1)

class ResidueGraphRequest(BaseModel):
    protein_id: str = Field(..., description="Protein identifier", example="ENSP00000327694")
    sequence: Optional[str] = Field(None, description="Protein sequence if not in database")
    
class NetworkResponse(BaseModel):
    nodes: List[Dict] = Field(..., description="List of protein nodes with metadata")
    edges: List[Dict] = Field(..., description="List of predicted or verified interactions")

class BatchPredictionRequest(BaseModel):
    pairs: List[ProteinPair] = Field(..., description="List of protein pairs to predict")

class MutationItem(BaseModel):
    protein: int = Field(..., description="Which protein in the pair to mutate (1 or 2)", example=1)
    pos: int = Field(..., description="1-indexed position of the mutation", example=152)
    orig: str = Field(..., description="Original amino acid", example="A")
    mut: str = Field(..., description="Mutated amino acid", example="V")

class MutationRequest(BaseModel):
    protein1_id: str = Field(..., example="ENSP00000327694")
    protein1_seq: Optional[str] = None
    protein2_id: str = Field(..., example="ENSP00000373627")
    protein2_seq: Optional[str] = None
    mutations: List[MutationItem] = Field(..., description="List of mutations to simulate")

class MutationResult(BaseModel):
    protein: int
    pos: int
    orig: str
    mut: str
    base_score: float = Field(..., description="Original interaction probability")
    mutated_score: float = Field(..., description="Interaction probability after mutation")
    impact_delta: float = Field(..., description="Change in probability (mutated - base)")
    interpretation: str = Field(..., description="Qualitative impact of the mutation")
    error: Optional[str] = None

class MutationAnalysisResponse(BaseModel):
    protein1: str
    protein2: str
    mutation_results: List[MutationResult]

class BioMetaResponse(BaseModel):
    protein_id: str
    uniprot_id: str
    localization: str = Field(..., description="Subcellular localization information")
    pathways: str = Field(..., description="Biological pathways involved")

class FeasibilityResponse(BaseModel):
    compatible: bool = Field(..., description="Whether the proteins can physically interact based on localization")
    intersection: List[str] = Field(..., description="Shared subcellular compartments")
    p1_locs: List[str]
    p2_locs: List[str]
    reason: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question about proteins or biology", example="What is p53?")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The assistant's answer in markdown format")
    suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    sources: List[str] = Field(default=[], description="Data sources used for the answer")

