import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.graph_model import GATLinkPredictor, GINLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR

def explain_prediction(protein1_id: str, protein2_id: str):
    device = torch.device("cpu")
    
    # ── Load Data ──────────────────────────────────────────────────
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    
    if not graph_data_path.exists() or not mapping_path.exists():
        return {"error": "Graph data or mapping missing"}

    data = torch.load(graph_data_path, weights_only=False).to(device)
    node_mapping = torch.load(mapping_path, weights_only=False)
    
    # ── Load Model ─────────────────────────────────────────────────
    model_path = MODELS_DIR / "graph_model_best.pth"
    if not model_path.exists():
        return {"error": f"Model not found at {model_path}"}
        
    state_dict = torch.load(model_path, map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    
    if is_gin:
        model = GINLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128).to(device)
    else:
        model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=256).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()

    # ── Prepare IDs ────────────────────────────────────────────────
    if protein1_id not in node_mapping or protein2_id not in node_mapping:
        return {"error": f"One or both proteins not in graph."}

    idx1 = node_mapping[protein1_id]
    idx2 = node_mapping[protein2_id]
    edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(device)

    # ── Setup Explainer ────────────────────────────────────────────
    # Scale epochs down for API speed (200 takes too long, 50 is enough for a summary)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='edge',
            return_type='raw',
        ),
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_label_index=edge_label_index
    )

    # ── Interpret Results ──────────────────────────────────────────
    results = {
        "top_features": [],
        "top_neighbors": []
    }
    
    # Feature Importance
    if 'node_mask' in explanation:
        node_feat_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
        top_indices = np.argsort(node_feat_importance)[-5:][::-1]
        for idx in top_indices:
            results["top_features"].append({"index": int(idx), "importance": float(node_feat_importance[idx])})

    # Edge Importance (Subgraphs)
    if 'edge_mask' in explanation:
        edge_importance = explanation.edge_mask.cpu().numpy()
        top_edges_idx = np.argsort(edge_importance)[-10:][::-1]
        
        inv_mapping = {v: k for k, v in node_mapping.items()}
        
        seen_neighbors = set()
        for e_idx in top_edges_idx:
            u, v = data.edge_index[:, e_idx]
            u_id, v_id = inv_mapping[u.item()], inv_mapping[v.item()]
            
            # We want to identify the "neighbor" of p1 or p2 that is influential
            neighbor_id = v_id if u_id in [protein1_id, protein2_id] else u_id
            
            if neighbor_id not in [protein1_id, protein2_id] and neighbor_id not in seen_neighbors:
                results["top_neighbors"].append({
                    "id": neighbor_id,
                    "importance": float(edge_importance[e_idx]),
                    "type": "bridge" if u_id not in [protein1_id, protein2_id] else "direct"
                })
                seen_neighbors.add(neighbor_id)
                if len(results["top_neighbors"]) >= 5: break

    return results

if __name__ == "__main__":
    # Example using two proteins from standard PPI sets if they exist
    # User can pass specific IDs as arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=str, default="ENSP00000269305") # Example TP53
    parser.add_argument("--p2", type=str, default="ENSP00000398846") # Example MDM2
    args = parser.parse_args()
    
    explain_prediction(args.p1, args.p2)
