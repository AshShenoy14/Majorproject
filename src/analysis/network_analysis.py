import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Any

class NetworkAnalyzer:
    def __init__(self, graph: nx.Graph = None):
        """
        Initialize the analyzer with a NetworkX graph.
        """
        self.graph = graph

    def build_from_dataframe(self, df: pd.DataFrame, source_col: str = 'protein1', target_col: str = 'protein2', weight_col: str = None):
        """
        Builds graph from a pandas DataFrame.
        """
        self.graph = nx.from_pandas_edgelist(df, source=source_col, target=target_col, edge_attr=weight_col)
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def calculate_centralities(self) -> pd.DataFrame:
        """
        Calculates Degree, Betweenness, and Closeness centrality.
        Returns a DataFrame sorted by Degree Centrality.
        """
        if self.graph is None or len(self.graph) == 0:
            return pd.DataFrame()

        print("Calculating Degree Centrality...")
        deg = nx.degree_centrality(self.graph)
        
        print("Calculating Betweenness Centrality...")
        # Approximation for large graphs
        k = 100 if len(self.graph) > 1000 else None
        bet = nx.betweenness_centrality(self.graph, k=k)
        
        print("Calculating Closeness Centrality...")
        clo = nx.closeness_centrality(self.graph)

        print("Calculating Eigenvector Centrality...")
        try:
            eig = nx.eigenvector_centrality(self.graph, max_iter=500)
        except:
            eig = {n: 0.0 for n in self.graph.nodes()} # Fallback if convergence fails
        
        data = []
        for node in self.graph.nodes():
            data.append({
                "protein_id": node,
                "degree_centrality": deg[node],
                "betweenness_centrality": bet[node],
                "closeness_centrality": clo[node],
                "eigenvector_centrality": eig[node]
            })
            
        df = pd.DataFrame(data)
        return df.sort_values("degree_centrality", ascending=False)

    def identify_hubs(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Identifies 'Hub' proteins based on high degree centrality.
        """
        if self.graph is None:
            return []
            
        deg = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)
        
        hubs = []
        for node, score in sorted_nodes[:top_k]:
            hubs.append({"id": node, "score": score, "type": "hub"})
            
        return hubs

    def visualize_top_hubs(self, top_k: int = 10, output_path: str = "top_hubs.png"):
        """
        Visualizes the subgraph of the top K hub proteins.
        """
        hubs = self.identify_hubs(top_k)
        if not hubs:
             print("No hubs to visualize.")
             return
             
        hub_ids = [h['id'] for h in hubs]
        subgraph = self.graph.subgraph(hub_ids)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph, seed=42)
        
        nx.draw_networkx_nodes(subgraph, pos, node_color='#ff9999', node_size=1200, edgecolors='black')
        nx.draw_networkx_edges(subgraph, pos, alpha=0.6, width=1.5)
        nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight="bold", font_family="sans-serif")
        
        plt.title(f"Protein Interaction Subgraph: Top {top_k} Hubs", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Hub visualization saved to {output_path}")

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Returns basic graph statistics.
        """
        if self.graph is None:
            return {}
            
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() < 2000 else "Skipped (Large Graph)"
        }
