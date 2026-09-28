import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import List, Literal

from app.backend import state
from app.backend.schemas import TherapeuticTargetResponse
from src.utils.paths import PROCESSED_DATA_DIR

router = APIRouter(tags=["Analysis"])

NetworkChoice = Query(
    "predicted",
    description="'predicted' = known training interactions + interactions the ensemble predicts on held-out pairs "
                "(scripts/build_predicted_network.py); 'known' = training interactions only. "
                "Falls back to 'known' if the predicted network has not been built.",
)


def _network_analyzer(network: str):
    """Returns (analyzer, network actually used)."""
    if network == "predicted" and "network_predicted" in state.analyzers:
        return state.analyzers["network_predicted"], "predicted"
    if "network" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Network Analysis not running (Check train.csv)")
    return state.analyzers["network"], "known"


@router.get("/network",
            summary="Get Verified Network Subgraph",
            description="Returns a subset of the positive interaction network for visualization.")
def get_network(limit: int = 100):
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
def get_centrality(top_k: int = 10, network: Literal["predicted", "known"] = NetworkChoice):
    """
    Returns top-K proteins by degree centrality in the interactome.
    """
    analyzer, used = _network_analyzer(network)

    try:
        df = analyzer.calculate_centralities()
        if df.empty:
            return []
        # Return top K nodes by Degree, with the PageRank stored in the graph node features
        records = df.head(top_k).to_dict(orient="records")
        pagerank = state.data_cache.get("pagerank", {})
        for rec in records:
            rec["pagerank"] = pagerank.get(rec["protein"])
            rec["network"] = used
        return records
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/therapeutic-targets",
            response_model=List[TherapeuticTargetResponse],
            summary="Get Prioritized Therapeutic Targets",
            description="Calculates Computational Therapeutic Target Priority Scores (TTPS) combining topological centrality and drug evidence.")
def get_therapeutic_targets(
    limit: int = 50,
    w_degree: float = 0.40,
    w_betweenness: float = 0.35,
    w_chembl: float = 0.25,
    network: Literal["predicted", "known"] = NetworkChoice,
):
    """
    Returns therapeutic target priorities ranked by TTPS score:
    TTPS = w_degree * NormDegree + w_betweenness * NormBetweenness + w_chembl * Indicator(ChEMBL Target)
    """
    analyzer, used = _network_analyzer(network)
    pred_counts = state.data_cache.get("predicted_edge_counts", {}) if used == "predicted" else {}
    novel_counts = state.data_cache.get("predicted_novel_counts", {}) if used == "predicted" else {}

    try:
        target_mgr = state.managers.get("target")
        df = analyzer.calculate_therapeutic_priority_score(
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
                "network": used,
                "predicted_interactions": int(pred_counts.get(str(r.get("protein_id", "")), 0)),
                "novel_predicted_interactions": int(novel_counts.get(str(r.get("protein_id", "")), 0)),
            })
        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/stats",
            summary="Get Network Statistics",
            description="Returns global statistics of the protein interaction network.")
def get_network_stats(network: Literal["predicted", "known"] = NetworkChoice):
    """
    Returns node count, edge count, and density metrics.
    """
    if "network" not in state.analyzers and "network_predicted" not in state.analyzers:
        return {}
    analyzer, used = _network_analyzer(network)
    stats = analyzer.get_graph_stats()
    stats["network"] = used
    stats["predicted_network_available"] = "network_predicted" in state.analyzers
    if used == "predicted":
        stats.update(state.data_cache.get("predicted_network_summary", {}))
    return stats
