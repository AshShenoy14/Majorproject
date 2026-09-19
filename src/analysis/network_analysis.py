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
        
        print("Calculating Betweenness Centrality (approx)...")
        k = 50 if len(self.graph) > 1000 else None
        bet = nx.betweenness_centrality(self.graph, k=k)
        
        print("Skipping Closeness Centrality for performance...")
        clo = {n: 0.0 for n in self.graph.nodes()}

        print("Calculating Eigenvector Centrality...")
        try:
            eig = nx.eigenvector_centrality(self.graph, max_iter=500)
        except:
            eig = {n: 0.0 for n in self.graph.nodes()} # Fallback if convergence fails
        
        data = []
        for node in self.graph.nodes():
            data.append({
                "protein": node, # Frontend: m.protein
                "degree": deg[node], # Frontend: m.degree
                "betweenness": bet[node], # Frontend: m.betweenness
                "protein_id": node,
                "degree_centrality": deg[node],
                "betweenness_centrality": bet[node],
                "closeness_centrality": clo[node],
                "eigenvector_centrality": eig[node]
            })
            
        df = pd.DataFrame(data)
        return df.sort_values("degree", ascending=False)

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

    def calculate_therapeutic_priority_score(
        self,
        target_manager = None,
        top_k: int = 50,
        w_degree: float = 0.40,
        w_betweenness: float = 0.35,
        w_chembl: float = 0.25
    ) -> pd.DataFrame:
        """
        Calculates the Computational Therapeutic Target Priority Score (TTPS) for proteins in the graph:
        TTPS = w_degree * NormDegree + w_betweenness * NormBetweenness + w_chembl * Indicator(ChEMBL Target)

        Weights default to 0.40, 0.35, 0.25 (sum = 1.0).
        NormDegree and NormBetweenness are min-max normalized floats in [0, 1].
        Indicator(ChEMBL Target) is 1.0 if known drug target / ChEMBL record exists, else 0.0.
        """
        df = self.calculate_centralities()
        if df.empty:
            return pd.DataFrame()

        # Min-max normalization for degree centrality
        deg_min = df['degree_centrality'].min()
        deg_max = df['degree_centrality'].max()
        deg_range = deg_max - deg_min
        if deg_range > 0:
            df['norm_degree'] = (df['degree_centrality'] - deg_min) / deg_range
        else:
            df['norm_degree'] = 0.0

        # Min-max normalization for betweenness centrality
        bet_min = df['betweenness_centrality'].min()
        bet_max = df['betweenness_centrality'].max()
        bet_range = bet_max - bet_min
        if bet_range > 0:
            df['norm_betweenness'] = (df['betweenness_centrality'] - bet_min) / bet_range
        else:
            df['norm_betweenness'] = 0.0

        # Ensure no NaNs / Infs
        df['norm_degree'] = df['norm_degree'].fillna(0.0).clip(0.0, 1.0)
        df['norm_betweenness'] = df['norm_betweenness'].fillna(0.0).clip(0.0, 1.0)

        # ChEMBL drug target lookup if target_manager is provided
        df['is_chembl_target'] = False
        df['chembl_id'] = None
        df['uniprot_id'] = "N/A"
        df['target_name'] = None
        df['target_type'] = None

        tm = target_manager if target_manager is not None else getattr(self, 'target_manager', None)
        if tm is not None:
            try:
                proteins_to_query = df['protein_id'].tolist()
                drug_targets_df = tm.get_targets(proteins_to_query)
                if not drug_targets_df.empty:
                    # Create lookup map
                    target_map = {}
                    for _, row in drug_targets_df.iterrows():
                        pid = row.get('protein_id')
                        if pid and pid not in target_map:
                            target_map[pid] = {
                                'chembl_id': row.get('chembl_id'),
                                'uniprot_id': row.get('uniprot_id', 'N/A'),
                                'target_name': row.get('target_name'),
                                'target_type': row.get('target_type'),
                            }

                    # Populate columns
                    for idx, row in df.iterrows():
                        pid = row['protein_id']
                        if pid in target_map:
                            df.at[idx, 'is_chembl_target'] = True
                            df.at[idx, 'chembl_id'] = target_map[pid]['chembl_id']
                            df.at[idx, 'uniprot_id'] = target_map[pid]['uniprot_id']
                            df.at[idx, 'target_name'] = target_map[pid]['target_name']
                            df.at[idx, 'target_type'] = target_map[pid]['target_type']
            except Exception as e:
                print(f"Warning: Drug target lookup failed during TTPS computation: {e}")

        # Compute TTPS
        chembl_val = df['is_chembl_target'].astype(float)
        df['ttps_score'] = (
            w_degree * df['norm_degree'] +
            w_betweenness * df['norm_betweenness'] +
            w_chembl * chembl_val
        )

        # NaN / Inf protection
        df['ttps_score'] = df['ttps_score'].fillna(0.0).clip(0.0, 1.0)

        # Deterministic sorting: desc(ttps_score), desc(degree_centrality), asc(protein_id)
        df = df.sort_values(
            by=['ttps_score', 'degree_centrality', 'protein_id'],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        df['rank'] = df.index + 1

        if top_k is not None:
            df = df.head(top_k)

        return df

