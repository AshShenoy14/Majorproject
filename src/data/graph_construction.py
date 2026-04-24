import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import os
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR

def construct_graph(interactions_df: pd.DataFrame, embeddings: Dict[str, torch.Tensor]) -> Tuple[Data, Dict[str, int]]:
    """
    Constructs a PyTorch Geometric Data object from interactions and embeddings.
    
    Args:
        interactions_df: DataFrame with 'protein1', 'protein2' columns (positive interactions).
        embeddings: Dictionary mapping ProteinID -> Embedding Tensor.
        
    Returns:
        data: PyG Data object.
        node_mapping: Dictionary mapping ProteinID -> Node Index.
    """
    print("Constructing graph...")
    
    # 1. Create Node Mapping
    # Get all unique proteins in the interactions
    proteins = set(interactions_df["protein1"]).union(set(interactions_df["protein2"]))
    
    # Filter proteins that have embeddings
    valid_proteins = [p for p in proteins if p in embeddings]
    if len(valid_proteins) < len(proteins):
        print(f"Warning: {len(proteins) - len(valid_proteins)} proteins missing embeddings.")
        
    node_mapping = {p: i for i, p in enumerate(valid_proteins)}
    
    # 2. Create Node Features (x)
    # Dimensionality from first embedding
    if not valid_proteins:
        print("Error: No valid proteins with embeddings found.")
        return Data(), {}
        
    emb_dim = embeddings[valid_proteins[0]].shape[0]
    num_nodes = len(valid_proteins)
    
    x = torch.empty((num_nodes, emb_dim), dtype=torch.float32)
    for p, i in node_mapping.items():
        x[i] = embeddings[p].float()
        
    # 3. Create Edge Index
    # Filter interactions where both proteins have embeddings
    valid_interactions = interactions_df[
        interactions_df["protein1"].isin(node_mapping) & 
        interactions_df["protein2"].isin(node_mapping)
    ]
    
    src = [node_mapping[p] for p in valid_interactions["protein1"]]
    dst = [node_mapping[p] for p in valid_interactions["protein2"]]
    
    # Undirected graph: add both directions
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    
    print(f"Graph constructed: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data, node_mapping

if __name__ == "__main__":
    # 1. Load Embeddings
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    if not emb_path.exists():
        print("Embeddings not found. Run feature_extraction.py first.")
        sys.exit(1)
        
    embeddings = torch.load(emb_path, weights_only=False)
    # Convert float16 embeddings to float32 for graph construction compatibility
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    
    # 2. Identify All Nodes (Train + Val + Test) to ensure consistent mapping
    print("Identifying all nodes...")
    all_proteins = set(embeddings.keys())
    
    # 3. Create Node Mapping (sorted for determinism)
    valid_proteins = sorted(list(all_proteins))
    node_mapping = {p: i for i, p in enumerate(valid_proteins)}
    
    # 4. Create Node Features
    emb_dim = embeddings[valid_proteins[0]].shape[0]
    num_nodes = len(valid_proteins)
    x = torch.empty((num_nodes, emb_dim), dtype=torch.float32)
    for p, i in node_mapping.items():
        x[i] = embeddings[p].float()
        
    # 5. Load Training Edges ONLY
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    # Filter for positive interactions only
    train_pos = train_df[train_df["label"] == 1]
    
    src = []
    dst = []
    for _, row in train_pos.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        if p1 in node_mapping and p2 in node_mapping:
            src.append(node_mapping[p1])
            dst.append(node_mapping[p2])
            
    # Undirected
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    # --- Advanced Structural Enrichment (90%+ Accuracy Strategy) ---
    print("Calculating topological features for network enrichment...")
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(src, dst))
    G.add_edges_from(edges)
    
    # Centrality metrics
    degree_cent = nx.degree_centrality(G)
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Convert to tensor (num_nodes, 3)
    topo_features = torch.zeros((num_nodes, 3), dtype=torch.float32)
    for i in range(num_nodes):
        topo_features[i, 0] = degree_cent.get(i, 0)
        topo_features[i, 1] = clustering.get(i, 0)
        topo_features[i, 2] = pagerank.get(i, 0)
    
    # Concatenate ESM embeddings with topological features
    x_enriched = torch.cat([x, topo_features], dim=-1)
    
    data = Data(x=x_enriched, edge_index=edge_index)
    print(f"Enriched Graph: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features.")
    
    # Clean up
    del embeddings
    gc.collect()
    
    # 6. Save Graph and Mapping
    torch.save(data, PROCESSED_DATA_DIR / "ppi_graph.pt")
    torch.save(node_mapping, PROCESSED_DATA_DIR / "ppi_graph_mapping.pt")
    print("Graph and mapping saved with structural enrichment.")
