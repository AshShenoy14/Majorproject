import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import sys
import os
import gc
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR

def construct_hetero_graph():
    print("Constructing Heterogeneous Graph...")
    
    # 1. Load Homogeneous PPI Graph and Mapping
    ppi_graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    
    if not ppi_graph_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Base PPI graph or mapping missing. Run graph_construction.py first.")
        
    homo_graph = torch.load(ppi_graph_path, weights_only=False)
    protein_mapping = torch.load(mapping_path, weights_only=False)
    
    # Initialize HeteroData
    data = HeteroData()
    
    # Set protein features (includes ESM-2 + topological features)
    data['protein'].x = homo_graph.x
    num_proteins = homo_graph.x.shape[0]
    
    # Keep track of mappings for other node types
    drug_mapping = {}
    disease_mapping = {}
    pathway_mapping = {}
    
    # Edge lists
    protein_protein_edges = homo_graph.edge_index
    drug_protein_edges = []
    disease_protein_edges = []
    pathway_protein_edges = []
    
    # Load metadata files
    bio_cache_path = PROCESSED_DATA_DIR / "bio_metadata_cache.csv"
    chembl_path = PROCESSED_DATA_DIR / "chembl_targets.csv"
    
    # 2. Extract Pathway Nodes and Edges
    if bio_cache_path.exists():
        bio_df = pd.read_csv(bio_cache_path).fillna("")
        for _, row in bio_df.iterrows():
            ensp = row["protein_id"]
            if ensp not in protein_mapping:
                continue
            prot_idx = protein_mapping[ensp]
            
            # Pathways
            p_str = str(row["pathways"]).strip()
            if p_str:
                pathways = [p.strip() for p in p_str.split(";") if p.strip()]
                for path in pathways:
                    if path not in pathway_mapping:
                        pathway_mapping[path] = len(pathway_mapping)
                    path_idx = pathway_mapping[path]
                    pathway_protein_edges.append((path_idx, prot_idx))
                    
    # 3. Extract Drug (ChEMBL Target) Nodes and Edges
    if chembl_path.exists():
        chembl_df = pd.read_csv(chembl_path).fillna("")
        for _, row in chembl_df.iterrows():
            ensp = row["protein_id"]
            if ensp not in protein_mapping:
                continue
            prot_idx = protein_mapping[ensp]
            
            chembl_id = str(row["chembl_id"]).strip()
            if chembl_id:
                if chembl_id not in drug_mapping:
                    drug_mapping[chembl_id] = len(drug_mapping)
                drug_idx = drug_mapping[chembl_id]
                drug_protein_edges.append((drug_idx, prot_idx))
                
    # 4. Extract/Generate Disease Nodes and Edges (Keyword-based Association)
    # Define a default set of common diseases to assign to proteins if keywords don't match
    common_diseases = [
        "Cancer", 
        "Alzheimer's Disease", 
        "Type 2 Diabetes", 
        "Cardiovascular Disease", 
        "Rheumatoid Arthritis"
    ]
    for d in common_diseases:
        disease_mapping[d] = len(disease_mapping)
        
    if bio_cache_path.exists():
        for _, row in bio_df.iterrows():
            ensp = row["protein_id"]
            if ensp not in protein_mapping:
                continue
            prot_idx = protein_mapping[ensp]
            
            families = str(row.get("families", "")).lower()
            domains = str(row.get("domains", "")).lower()
            loc = str(row.get("localization", "")).lower()
            
            assigned = False
            # Check Cancer keywords
            if any(k in families or k in domains or k in loc for k in ["tumor", "cancer", "kinase", "oncogene", "p53", "cell cycle"]):
                disease_protein_edges.append((disease_mapping["Cancer"], prot_idx))
                assigned = True
            # Check Neurodegenerative (Alzheimer's)
            if any(k in families or k in domains or k in loc for k in ["amyloid", "tau", "neuron", "brain", "synapse", "myelin"]):
                disease_protein_edges.append((disease_mapping["Alzheimer's Disease"], prot_idx))
                assigned = True
            # Check Diabetes/Metabolic
            if any(k in families or k in domains or k in loc for k in ["insulin", "glucose", "glyco", "metab", "lipid"]):
                disease_protein_edges.append((disease_mapping["Type 2 Diabetes"], prot_idx))
                assigned = True
            # Check Cardiovascular
            if any(k in families or k in domains or k in loc for k in ["heart", "cardiac", "muscle", "vessel", "blood"]):
                disease_protein_edges.append((disease_mapping["Cardiovascular Disease"], prot_idx))
                assigned = True
            # Check Immune (Rheumatoid Arthritis)
            if any(k in families or k in domains or k in loc for k in ["immune", "rheuma", "arthritis", "inflamm", "interleukin", "cytokine"]):
                disease_protein_edges.append((disease_mapping["Rheumatoid Arthritis"], prot_idx))
                assigned = True
                
            # Fallback: assign to a disease deterministically based on hash
            if not assigned:
                fallback_disease = common_diseases[hash(ensp) % len(common_diseases)]
                disease_protein_edges.append((disease_mapping[fallback_disease], prot_idx))
                
    # 5. Build Edge Indices
    # protein - interacts_with - protein
    data['protein', 'interacts_with', 'protein'].edge_index = protein_protein_edges
    
    # drug - targets - protein (and reverse)
    if drug_protein_edges:
        drug_prot_tensor = torch.tensor(drug_protein_edges, dtype=torch.long).t()
        data['drug', 'targets', 'protein'].edge_index = drug_prot_tensor
        data['protein', 'targeted_by', 'drug'].edge_index = torch.stack([drug_prot_tensor[1], drug_prot_tensor[0]])
    else:
        # Create empty edges to prevent PyG errors
        data['drug', 'targets', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['protein', 'targeted_by', 'drug'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # disease - associated_with - protein (and reverse)
    if disease_protein_edges:
        dis_prot_tensor = torch.tensor(disease_protein_edges, dtype=torch.long).t()
        data['disease', 'associated_with', 'protein'].edge_index = dis_prot_tensor
        data['protein', 'associated_with', 'disease'].edge_index = torch.stack([dis_prot_tensor[1], dis_prot_tensor[0]])
    else:
        data['disease', 'associated_with', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['protein', 'associated_with', 'disease'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # pathway - contains - protein (and reverse)
    if pathway_protein_edges:
        path_prot_tensor = torch.tensor(pathway_protein_edges, dtype=torch.long).t()
        data['pathway', 'contains', 'protein'].edge_index = path_prot_tensor
        data['protein', 'participates_in', 'pathway'].edge_index = torch.stack([path_prot_tensor[1], path_prot_tensor[0]])
    else:
        data['pathway', 'contains', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['protein', 'participates_in', 'pathway'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # 6. Initialize Node Features
    emb_dim = 128
    
    # Initialize Drug Features
    num_drugs = max(1, len(drug_mapping))
    data['drug'].x = torch.randn((num_drugs, emb_dim))
    
    # Initialize Disease Features
    num_diseases = max(1, len(disease_mapping))
    data['disease'].x = torch.randn((num_diseases, emb_dim))
    
    # Initialize Pathway Features
    num_pathways = max(1, len(pathway_mapping))
    data['pathway'].x = torch.randn((num_pathways, emb_dim))
    
    # Print status
    print("Heterogeneous Graph constructed:")
    for ntype in data.node_types:
        print(f"  Node type '{ntype}': {data[ntype].num_nodes} nodes, {data[ntype].num_features} features")
    for etype in data.edge_types:
        print(f"  Edge type '{etype}': {data[etype].num_edges} edges")
        
    # Save graph and mapping mappings
    torch.save(data, PROCESSED_DATA_DIR / "ppi_hetero_graph.pt")
    
    mappings = {
        "protein": protein_mapping,
        "drug": drug_mapping,
        "disease": disease_mapping,
        "pathway": pathway_mapping
    }
    torch.save(mappings, PROCESSED_DATA_DIR / "ppi_hetero_mappings.pt")
    print("Saved heterogeneous graph and mappings.")
    return data

if __name__ == "__main__":
    construct_hetero_graph()
