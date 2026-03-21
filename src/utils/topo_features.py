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
