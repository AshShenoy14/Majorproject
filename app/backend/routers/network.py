import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import List

from app.backend import state
from app.backend.schemas import TherapeuticTargetResponse
from src.utils.paths import PROCESSED_DATA_DIR

router = APIRouter(tags=["Analysis"])


@router.get("/network",
            summary="Get Verified Network Subgraph",
            description="Returns a subset of the positive interaction network for visualization.")
async def get_network(limit: int = 100):
    """
    Fetches the top N verified interactions from the training set.
    """

    try:
        train_path = PROCESSED_DATA_DIR / "train.csv"
        if not train_path.exists():
            return {"nodes": [], "edges": []}

        df = pd.read_csv(train_path)
        df = df[df["label"] == 1].head(limit)

        nodes = set()
        edges = []

        for _, row in df.iterrows():
            p1, p2 = row["protein1"], row["protein2"]
            nodes.add(p1)
            nodes.add(p2)
            edges.append({"source": p1, "target": p2, "weight": 1.0, "type": "verified"})

        node_list = [{"id": n, "label": n} for n in nodes]

        return {"nodes": node_list, "edges": edges}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/centrality",
            summary="Get Network Centrality",
            description="Calculates node centrality metrics for proteins in the interaction network.")
async def get_centrality(top_k: int = 10):
    """
    Returns top-K proteins by degree centrality in the interactome.
    """

    if "network" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Network Analysis not running (Check train.csv)")

    try:
        df = state.analyzers["network"].calculate_centralities()
        if df.empty:
            return []
        # Return top K nodes by Degree, with the PageRank stored in the graph node features
        records = df.head(top_k).to_dict(orient="records")
        pagerank = state.data_cache.get("pagerank", {})
        for rec in records:
            rec["pagerank"] = pagerank.get(rec["protein"])
        return records
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/therapeutic-targets",
            response_model=List[TherapeuticTargetResponse],
            summary="Get Prioritized Therapeutic Targets",
            description="Calculates Computational Therapeutic Target Priority Scores (TTPS) combining topological centrality and drug evidence.")
async def get_therapeutic_targets(
    limit: int = 50,
    w_degree: float = 0.40,
    w_betweenness: float = 0.35,
    w_chembl: float = 0.25
):
    """
    Returns therapeutic target priorities ranked by TTPS score:
    TTPS = w_degree * NormDegree + w_betweenness * NormBetweenness + w_chembl * Indicator(ChEMBL Target)
    """
    if "network" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Network Analysis not running (Check train.csv)")

    try:
        target_mgr = state.managers.get("target")
        df = state.analyzers["network"].calculate_therapeutic_priority_score(
            target_manager=target_mgr,
            top_k=limit,
            w_degree=w_degree,
            w_betweenness=w_betweenness,
            w_chembl=w_chembl
        )
        if df.empty:
            return []

        records = df.to_dict(orient="records")
        results = []
        for r in records:
            results.append({
                "rank": int(r.get("rank", 0)),
                "protein_id": str(r.get("protein_id", "")),
                "uniprot_id": str(r.get("uniprot_id") or "N/A"),
                "ttps_score": float(r.get("ttps_score", 0.0)),
                "norm_degree": float(r.get("norm_degree", 0.0)),
                "norm_betweenness": float(r.get("norm_betweenness", 0.0)),
                "is_chembl_target": bool(r.get("is_chembl_target", False)),
                "chembl_id": r.get("chembl_id") if pd.notna(r.get("chembl_id")) else None,
                "target_name": r.get("target_name") if pd.notna(r.get("target_name")) else None,
                "target_type": r.get("target_type") if pd.notna(r.get("target_type")) else None,
                "degree_centrality": float(r.get("degree_centrality", 0.0)),
                "betweenness_centrality": float(r.get("betweenness_centrality", 0.0)),
                "eigenvector_centrality": float(r.get("eigenvector_centrality", 0.0)),
            })
        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/stats",
            summary="Get Network Statistics",
            description="Returns global statistics of the protein interaction network.")
async def get_network_stats():
    """
    Returns node count, edge count, and density metrics.
    """
    if "network" not in state.analyzers:
        return {}
    return state.analyzers["network"].get_graph_stats()
