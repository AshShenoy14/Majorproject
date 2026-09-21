import re
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional, Dict

AMINO_ACID_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

class ProteinPair(BaseModel):
    protein1_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 1", json_schema_extra={"example": "MAH..."})
    protein2_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 2", json_schema_extra={"example": "MVK..."})
    protein1_id: Optional[str] = Field(None, description="Identifier for protein 1 (e.g., ENSP ID)", json_schema_extra={"example": "ENSP00000327694"})
    protein2_id: Optional[str] = Field(None, description="Identifier for protein 2 (e.g., ENSP ID)", json_schema_extra={"example": "ENSP00000373627"})

    @field_validator("protein1_seq", "protein2_seq")
    @classmethod
    def validate_amino_acid_sequence(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        cleaned = re.sub(r"\s+", "", v).upper()
        if not cleaned:
            return None
        if len(cleaned) < 5:
            raise ValueError("Protein sequence must be at least 5 amino acids long.")
        if len(cleaned) > 2000:
            raise ValueError("Protein sequence exceeds maximum supported length of 2000 residues.")
        if not AMINO_ACID_PATTERN.match(cleaned):
            invalid_chars = sorted(list(set(cleaned) - set("ACDEFGHIKLMNPQRSTVWY")))
            raise ValueError(f"Invalid amino acid character(s): {invalid_chars}. Only standard 20 amino acids allowed.")
        return cleaned

class PredictionResponse(BaseModel):
    protein1_id: Optional[str] = Field(None, description="Identifier for protein 1")
    protein2_id: Optional[str] = Field(None, description="Identifier for protein 2")
    status: str = Field("success", description="Prediction status: 'success' or 'error'")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    interaction_probability: Optional[float] = Field(None, description="Final ensemble prediction probability", json_schema_extra={"example": 0.88})
    esm_probability: Optional[float] = Field(None, description="Probability from the ESM-MLP sequence model", json_schema_extra={"example": 0.92})
    gat_probability: Optional[float] = Field(None, description="Probability from the GraphSAGE graph model (field name kept as gat_probability for API compatibility)", json_schema_extra={"example": 0.75})
    confidence_score: Optional[float] = Field(None, description="Normalized confidence score [0, 1]", json_schema_extra={"example": 0.76})
    explanation: Optional[Dict[str, Any]] = Field(None, description="Feature importance scores (e.g., SHAP values)")
    shap_explanations: Optional[List[float]] = Field(None, description="Direct SHAP attribution array for easy UI consumption")
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
    intersection: List[str] = Field(default=[], description="Shared subcellular compartments")
    p1_locs: List[str] = Field(default=[], description="Protein 1 subcellular compartments")
    p2_locs: List[str] = Field(default=[], description="Protein 2 subcellular compartments")
    reason: str = Field(default="", description="Biological rationale for compatibility assessment")

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

class TherapeuticTargetResponse(BaseModel):
    rank: int = Field(..., description="Target rank order")
    protein_id: str = Field(..., description="Protein identifier (ENSP)")
    uniprot_id: Optional[str] = Field("N/A", description="Mapped UniProt ID")
    ttps_score: float = Field(..., description="Computational Therapeutic Target Priority Score [0.0 - 1.0]")
    norm_degree: float = Field(..., description="Normalized Degree Centrality [0.0 - 1.0]")
    norm_betweenness: float = Field(..., description="Normalized Betweenness Centrality [0.0 - 1.0]")
    is_chembl_target: bool = Field(..., description="Whether protein has verified ChEMBL drug target evidence")
    chembl_id: Optional[str] = Field(None, description="ChEMBL target identifier")
    target_name: Optional[str] = Field(None, description="ChEMBL target preferred name")
    target_type: Optional[str] = Field(None, description="ChEMBL target classification type")
    degree_centrality: float = Field(..., description="Raw degree centrality in PPI network")
    betweenness_centrality: float = Field(..., description="Raw betweenness centrality in PPI network")
    eigenvector_centrality: float = Field(..., description="Raw eigenvector centrality in PPI network")

