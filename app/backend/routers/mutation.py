from fastapi import APIRouter, HTTPException

from app.backend import state
from app.backend.schemas import MutationRequest, MutationAnalysisResponse, ProteinPair, ResidueGraphRequest

router = APIRouter(tags=["Analysis"])


@router.post("/analysis/mutate",
             response_model=MutationAnalysisResponse,
             summary="Analyze Mutation Impact",
             description="Predicts how specific amino acid mutations affect the interaction probability of a protein pair.")
async def scan_mutations(request: MutationRequest):
    """
    Simulates in-silico mutations and measures the delta in interaction probability.
    """

    if "mutation" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Mutation Analyzer not initialized (Check models)")

    try:
        # Get sequences if missing
        sequences = {}
        to_fetch = []
        p1_id, p1_seq = request.protein1_id, request.protein1_seq
        p2_id, p2_seq = request.protein2_id, request.protein2_seq

        if not p1_seq: to_fetch.append(p1_id)
        else: sequences[p1_id] = p1_seq

        if not p2_seq: to_fetch.append(p2_id)
        else: sequences[p2_id] = p2_seq

        if to_fetch:
            sequences.update(state.managers["sequence"].get_sequences(to_fetch))

        if p1_id not in sequences or p2_id not in sequences:
            raise HTTPException(status_code=404, detail="Could not find sequences for one or both proteins.")

        results = state.analyzers["mutation"].project_mutation_impact(
            p1_id, sequences[p1_id],
            p2_id, sequences[p2_id],
            [m.dict() for m in request.mutations]
        )

        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/hotspots",
             summary="Identify Interaction Hotspots",
             description="Detects critical residues (hotspots) for the interaction using gradient-based importance.")
async def get_hotspots(request: ProteinPair):
    """
    Identifies specific amino acids that contribute most significantly to the interaction.
    """

    if "hotspot" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Hotspot Analyzer not initialized")

    p1 = request.protein1_id.strip() if request.protein1_id else None
    p2 = request.protein2_id.strip() if request.protein2_id else None
    if not p1 or not p2:
        raise HTTPException(status_code=400, detail="Both protein1_id and protein2_id are required.")

    try:
        # Get sequences if missing
        sequences = {}
        to_fetch = []
        if not request.protein1_seq: to_fetch.append(p1)
        else: sequences[p1] = request.protein1_seq
        if not request.protein2_seq: to_fetch.append(p2)
        else: sequences[p2] = request.protein2_seq

        if to_fetch:
            sequences.update(state.managers["sequence"].get_sequences(to_fetch))

        if p1 not in sequences or p2 not in sequences:
            raise HTTPException(status_code=404, detail="Could not find sequences for one or both proteins.")

        return state.analyzers["hotspot"].identify_hotspots(
            p1, sequences[p1],
            p2, sequences[p2]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/explain-gnn",
             summary="Deep GNN Feature and Neighbor Explanation",
             description="Runs PyTorch Geometric GNNExplainer on a pair for detailed research explainability.")
async def explain_gnn_deep(request: ProteinPair):
    from src.analysis.explain_model import explain_prediction as explain_gnn

    p1 = request.protein1_id.strip() if request.protein1_id else None
    p2 = request.protein2_id.strip() if request.protein2_id else None
    if not p1 or not p2:
        raise HTTPException(status_code=400, detail="Both protein1_id and protein2_id are required.")

    res_p1 = state.managers["id_mapper"].resolve_to_graph_id(p1, set(state.data_cache.get("mapping", {}).keys()))
    res_p2 = state.managers["id_mapper"].resolve_to_graph_id(p2, set(state.data_cache.get("mapping", {}).keys()))

    if "mapping" not in state.data_cache or res_p1 not in state.data_cache["mapping"] or res_p2 not in state.data_cache["mapping"]:
        raise HTTPException(status_code=404, detail="One or both proteins not found in the graph network.")

    try:
        return explain_gnn(
            res_p1, res_p2,
            model=state.models.get("graph_model"),
            data=state.data_cache.get("graph"),
            node_mapping=state.data_cache.get("mapping"),
            epochs=15
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GNNExplainer failed: {e}")


@router.post("/analysis/residue_graph",
             summary="Generate Residue Interaction Graph",
             description="Generates a graph representation of internal residue-residue interactions for a single protein.")
async def get_residue_graph(request: ResidueGraphRequest):
    """
    Constructs a RIG (Residue Interaction Graph) based on distance-truncated contacts.
    """

    if "residue_graph" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Residue Graph Generator not initialized")

    try:
        protein_id = request.protein_id
        sequence = request.sequence

        if not sequence:
            seq_dict = state.managers["sequence"].get_sequences([protein_id])
            if protein_id not in seq_dict:
                raise HTTPException(status_code=404, detail="Sequence not found")
            sequence = seq_dict[protein_id]

        uniprot_id = None
        if "id_mapper" in state.managers:
            uniprot_maps = state.managers["id_mapper"].ensp_to_uniprot([protein_id])
            uniprot_id = uniprot_maps.get(protein_id)

        return state.analyzers["residue_graph"].generate_rig(sequence, uniprot_id=uniprot_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/optimize",
             summary="Optimize Interaction",
             description="Suggests mutations to either disrupt or enhance a protein-protein interaction.")
async def get_optimization(request: ProteinPair, mode: str = "disrupt"):
    """
    In-silico optimization or disruption of a PPI.
    """

    if "mutation" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Mutation Analyzer not initialized")

    try:
        # Get sequences
        sequences = {}
        to_fetch = []
        if not request.protein1_seq: to_fetch.append(request.protein1_id)
        else: sequences[request.protein1_id] = request.protein1_seq
        if not request.protein2_seq: to_fetch.append(request.protein2_id)
        else: sequences[request.protein2_id] = request.protein2_seq

        if to_fetch:
            sequences.update(state.managers["sequence"].get_sequences(to_fetch))

        return state.analyzers["mutation"].suggest_optimal_mutations(
            request.protein1_id, sequences[request.protein1_id],
            request.protein2_id, sequences[request.protein2_id],
            mode=mode
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
