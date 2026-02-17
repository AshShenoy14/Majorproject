import networkx as nx
import pandas as pd
from typing import List, Tuple

def build_network_from_predictions(interactions_df: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    """
    Builds a NetworkX graph from predicted interactions.
    Args:
        interactions_df: DataFrame with 'protein1', 'protein2', 'score'.
        threshold: Minimum score to consider an edge.
    """
    G = nx.Graph()
    for _, row in interactions_df.iterrows():
        if row["score"] >= threshold:
            G.add_edge(row["protein1"], row["protein2"], weight=row["score"])
    return G

def calculate_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Calculates Degree, Betweenness, and Closeness centrality.
    Returns a DataFrame sorted by Degree Centrality.
    """
    print("Calculating Degree Centrality...")
    deg = nx.degree_centrality(G)
    
    print("Calculating Betweenness Centrality (this may take time)...")
    # For large graphs, k should be used for approximation
    bet = nx.betweenness_centrality(G, k=100 if len(G) > 1000 else None)
    
    print("Calculating Closeness Centrality...")
    clo = nx.closeness_centrality(G)
    
    data = []
    for node in G.nodes():
        data.append({
            "Protein": node,
            "Degree": deg[node],
            "Betweenness": bet[node],
            "Closeness": clo[node]
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values("Degree", ascending=False)
    return df

if __name__ == "__main__":
    # Example
    # G = nx.erdos_renyi_graph(100, 0.05)
    # df = calculate_centralities(G)
    # print(df.head())
    pass
