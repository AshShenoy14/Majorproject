import pytest
import networkx as nx
import pandas as pd
from src.analysis.network_analysis import NetworkAnalyzer

def test_calculate_therapeutic_priority_score():
    # Build a small dummy graph
    G = nx.Graph()
    G.add_edge("P01", "P02", weight=0.9)
    G.add_edge("P01", "P03", weight=0.8)
    G.add_edge("P02", "P03", weight=0.7)
    G.add_edge("P02", "P04", weight=0.95)

    analyzer = NetworkAnalyzer()
    analyzer.graph = G

    # TargetManager check mock using get_targets
    class DummyTargetManager:
        def get_targets(self, protein_ids):
            data = []
            for pid in protein_ids:
                if pid == "P01":
                    data.append({
                        "protein_id": "P01",
                        "chembl_id": "CHEMBL123",
                        "uniprot_id": "P01",
                        "target_name": "Test Target 1",
                        "target_type": "SINGLE PROTEIN"
                    })
            return pd.DataFrame(data)

    analyzer.target_manager = DummyTargetManager()

    df = analyzer.calculate_therapeutic_priority_score(
        w_degree=0.40,
        w_betweenness=0.35,
        w_chembl=0.25
    )

    assert len(df) == 4
    # Ensure items are ordered descending by ttps_score
    scores = df["ttps_score"].tolist()
    assert scores == sorted(scores, reverse=True)

    # Ensure required columns exist
    required_cols = [
        "protein_id", "rank", "ttps_score", "norm_degree",
        "norm_betweenness", "is_chembl_target", "degree_centrality",
        "betweenness_centrality"
    ]
    for col in required_cols:
        assert col in df.columns

    # Verify score bounds
    assert (df["ttps_score"] >= 0.0).all() and (df["ttps_score"] <= 1.0).all()

    # Check P01 gets the ChEMBL bonus
    p01_row = df[df["protein_id"] == "P01"].iloc[0]
    assert p01_row["is_chembl_target"] == True
    assert p01_row["chembl_id"] == "CHEMBL123"

