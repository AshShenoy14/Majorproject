from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict

class ProteinPair(BaseModel):
    protein1_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 1", json_schema_extra={"example": "MAH..."})
    protein2_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 2", json_schema_extra={"example": "MVK..."})
    protein1_id: Optional[str] = Field(None, description="Identifier for protein 1 (e.g., ENSP ID)", json_schema_extra={"example": "ENSP00000327694"})
    protein2_id: Optional[str] = Field(None, description="Identifier for protein 2 (e.g., ENSP ID)", json_schema_extra={"example": "ENSP00000373627"})

class PredictionResponse(BaseModel):
    interaction_probability: float = Field(..., description="Final ensemble prediction probability", json_schema_extra={"example": 0.88})
    esm_probability: float = Field(..., description="Probability from the ESM-MLP sequence model", json_schema_extra={"example": 0.92})
    gat_probability: float = Field(..., description="Probability from the GAT graph model", json_schema_extra={"example": 0.75})
    confidence_score: float = Field(..., description="Normalized confidence score [0, 1]", json_schema_extra={"example": 0.76})
    explanation: Dict[str, Any] = Field(..., description="Feature importance scores (e.g., SHAP values)")
    gnn_explanation: Optional[Dict[str, Any]] = Field(None, description="Detailed GNN-specific neighbor importance")
    protein1_uniprot_id: Optional[str] = Field(None, description="Mapped UniProt ID for protein 1", json_schema_extra={"example": "P12345"})
    protein2_uniprot_id: Optional[str] = Field(None, description="Mapped UniProt ID for protein 2", json_schema_extra={"example": "Q67890"})
    protein1_seq: Optional[str] = Field(None, description="Sequence used for protein 1")
    protein2_seq: Optional[str] = Field(None, description="Sequence used for protein 2")
    
class NetworkRequest(BaseModel):
    threshold: float = Field(0.5, description="Interaction probability threshold for network inclusion", ge=0, le=1)

class ResidueGraphRequest(BaseModel):
    protein_id: str = Field(..., description="Protein identifier", json_schema_extra={"example": "ENSP00000327694"})
    sequence: Optional[str] = Field(None, description="Protein sequence if not in database")
    
class NetworkResponse(BaseModel):
    nodes: List[Dict] = Field(..., description="List of protein nodes with metadata")
    edges: List[Dict] = Field(..., description="List of predicted or verified interactions")

class BatchPredictionRequest(BaseModel):
    pairs: List[ProteinPair] = Field(..., description="List of protein pairs to predict")

class MutationItem(BaseModel):
    protein: int = Field(..., description="Which protein in the pair to mutate (1 or 2)", json_schema_extra={"example": 1})
    pos: int = Field(..., description="1-indexed position of the mutation", json_schema_extra={"example": 152})
    orig: str = Field(..., description="Original amino acid", json_schema_extra={"example": "A"})
    mut: str = Field(..., description="Mutated amino acid", json_schema_extra={"example": "V"})

class MutationRequest(BaseModel):
    protein1_id: str = Field(..., json_schema_extra={"example": "ENSP00000327694"})
    protein1_seq: Optional[str] = None
    protein2_id: str = Field(..., json_schema_extra={"example": "ENSP00000373627"})
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
    is_in_interaction_region: Optional[bool] = Field(None, description="Whether mutation is in a predicted interaction region")
    interaction_region: Optional[str] = Field(None, description="Region label if in interaction region")
    error: Optional[str] = None

class IRLMRequest(BaseModel):
    protein1_id: Optional[str] = Field(None, description="Identifier for protein 1", json_schema_extra={"example": "ENSP00000327694"})
    protein2_id: Optional[str] = Field(None, description="Identifier for protein 2", json_schema_extra={"example": "ENSP00000373627"})
    protein1_seq: Optional[str] = Field(None, description="Sequence for protein 1")
    protein2_seq: Optional[str] = Field(None, description="Sequence for protein 2")
    base_probability: float = Field(0.5, description="Base interaction probability")

class InteractionRegion(BaseModel):
    start: int
    end: int
    score: float
    sequence_snippet: str

class IRLMResponse(BaseModel):
    protein1_regions: List[InteractionRegion]
    protein2_regions: List[InteractionRegion]
    protein1_hotspots: List[int]
    protein2_hotspots: List[int]
    attention_map_shape: List[int]
    protein_A_region: Optional[List[int]] = None
    protein_B_region: Optional[List[int]] = None
    protein_A_importance_scores: Optional[List[float]] = None
    protein_B_importance_scores: Optional[List[float]] = None
    top_residue_pairs: Optional[List[Dict[str, Any]]] = None
    hotspot_residues: Optional[List[Dict[str, Any]]] = None
    region_score: Optional[float] = None
    region_confidence: Optional[float] = None

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
    message: str = Field(..., description="User's question about proteins or biology", json_schema_extra={"example": "What is p53?"})

class ChatResponse(BaseModel):
    response: str = Field(..., description="The assistant's answer in markdown format")
    suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    sources: List[str] = Field(default=[], description="Data sources used for the answer")

class HeteroPredictionRequest(BaseModel):
    protein1_id: str = Field(..., description="ID for protein 1 (ENSP)", json_schema_extra={"example": "ENSP00000327694"})
    protein2_id: str = Field(..., description="ID for protein 2 (ENSP)", json_schema_extra={"example": "ENSP00000373627"})
    protein1_seq: Optional[str] = Field(None, description="Sequence for protein 1 if not in database")
    protein2_seq: Optional[str] = Field(None, description="Sequence for protein 2 if not in database")
    context_drugs: Optional[List[str]] = Field(default=[], description="List of drug IDs (CHEMBL) or names to check for connections", json_schema_extra={"example": ["CHEMBL2918"]})
    context_diseases: Optional[List[str]] = Field(default=[], description="List of disease names to check for connections", json_schema_extra={"example": ["Cancer"]})
    context_pathways: Optional[List[str]] = Field(default=[], description="List of pathway names to check for connections", json_schema_extra={"example": []})

class HeteroPredictionResponse(BaseModel):
    interaction_probability: float = Field(..., description="Interaction probability predicted by HeteroGNN", json_schema_extra={"example": 0.87})
    confidence_score: float = Field(..., description="Normalized confidence score [0, 1]", json_schema_extra={"example": 0.74})
    protein1_id: str
    protein2_id: str
    protein1_context_connections: Dict[str, List[str]] = Field(..., description="Connections for protein 1 to context nodes")
    protein2_context_connections: Dict[str, List[str]] = Field(..., description="Connections for protein 2 to context nodes")
    shared_context: List[str] = Field(..., description="Shared context nodes connected to both proteins")
