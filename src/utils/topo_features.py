import networkx as nx
import torch
import pandas as pd
from src.utils.paths import PROCESSED_DATA_DIR

class TopologicalFeatureExtractor:
    def __init__(self, edges_df):
        """
        Extracts complex structural features (Topology) from a PPI Graph.
        """
        self.G = nx.from_pandas_edgelist(edges_df, "protein1", "protein2")
        print(f"Graph initialized with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

    def get_features(self, proteins_list):
        """
        Calculates node-level topological features:
        1. Pagerank (Universal Importance)
        2. Degree Centrality (Local Connectivity)
        3. Hub Score (HITS)
        4. Authority Score (HITS)
        """
        print("Calculating PageRank (Importance)...")
        pagerank = nx.pagerank(self.G)
        
        print("Calculating Degree Centrality (Popularity)...")
        degree = nx.degree_centrality(self.G)
        
        print("Calculating HITS (Hubs & Authorities)...")
        try:
            hubs, authorities = nx.hits(self.G, max_iter=50)
        except:
            print("  HITS failed to converge, using zeros.")
            hubs = {p: 0.0 for p in self.G.nodes()}
            authorities = {p: 0.0 for p in self.G.nodes()}

        features = {}
        for p in proteins_list:
            if p in self.G.nodes():
                features[p] = torch.tensor([
                    pagerank.get(p, 0.0),
                    degree.get(p, 0.0),
                    hubs.get(p, 0.0),
                    authorities.get(p, 0.0)
                ], dtype=torch.float32)
            else:
                features[p] = torch.zeros(4, dtype=torch.float32)
                
        print(f"Extracted 4 topological features for {len(proteins_list)} proteins.")
        return features


def node_topology_columns(src, dst, num_nodes):
    """
    Per-node structural columns (degree centrality, clustering coefficient, PageRank a=0.85) computed on the
    graph defined by the undirected edge list (src[i], dst[i]) over `num_nodes` nodes.

    Same recipe as src/data/graph_construction.py. Called once per OOF fold on that fold's TRAINING edges only,
    so held-out pairs never influence a fold's node features.
    Returns a float32 tensor of shape (num_nodes, 3).
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(list(src), list(dst)))
    dc = nx.degree_centrality(G)
    cl = nx.clustering(G)
    pr = nx.pagerank(G, alpha=0.85)
    cols = torch.zeros((num_nodes, 3), dtype=torch.float32)
    for i in range(num_nodes):
        cols[i, 0], cols[i, 1], cols[i, 2] = dc.get(i, 0), cl.get(i, 0), pr.get(i, 0)
    return cols
