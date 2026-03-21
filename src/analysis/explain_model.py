import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def explain_prediction(protein1_id: str, protein2_id: str):
    device = torch.device("cpu") # Explainer is often easier to debug on CPU
    
    # ── Load Data ──────────────────────────────────────────────────
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    
    if not graph_data_path.exists() or not mapping_path.exists():
        print("Graph or mapping files missing.")
        return

    data = torch.load(graph_data_path, weights_only=False).to(device)
    node_mapping = torch.load(mapping_path, weights_only=False)
    
    # ── Load Model ─────────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    model = GATLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128, heads=4).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Prepare IDs ────────────────────────────────────────────────
    if protein1_id not in node_mapping or protein2_id not in node_mapping:
        print(f"One or both proteins not in graph: {protein1_id}, {protein2_id}")
        return

    idx1 = node_mapping[protein1_id]
    idx2 = node_mapping[protein2_id]
    edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(device)

    # ── Setup Explainer ────────────────────────────────────────────
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='edge',
            return_type='logits',
        ),
    )

    print(f"Explaining interaction between {protein1_id} and {protein2_id}...")
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_label_index=edge_label_index
    )

    # ── Interpret Results ──────────────────────────────────────────
    print("\n=== Explanation Summary ===")
    
    # Feature Importance (Top 10)
    if 'node_mask' in explanation:
        node_feat_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
        top_indices = np.argsort(node_feat_importance)[-10:][::-1]
        print("\nTop 10 Influential Protein Feature Indices:")
        for idx in top_indices:
            print(f"  Index {idx:3d}: Importance {node_feat_importance[idx]:.4f}")

    # Edge Importance (Subgraphs)
    if 'edge_mask' in explanation:
        edge_importance = explanation.edge_mask.cpu().numpy()
        top_edges = np.argsort(edge_importance)[-5:][::-1]
        print("\nTop 5 Supporting Network Edges (Proteins):")
        
        # Reverse mapping for display
        inv_mapping = {v: k for k, v in node_mapping.items()}
        
        for e_idx in top_edges:
            u, v = data.edge_index[:, e_idx]
            u_id, v_id = inv_mapping[u.item()], inv_mapping[v.item()]
            print(f"  {u_id} <-> {v_id}: Importance {edge_importance[e_idx]:.4f}")

    print("\nExplainment complete. These features and neighbors are the primary drivers of the PPI prediction.")

if __name__ == "__main__":
    # Example using two proteins from standard PPI sets if they exist
    # User can pass specific IDs as arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=str, default="ENSP00000269305") # Example TP53
    parser.add_argument("--p2", type=str, default="ENSP00000398846") # Example MDM2
    args = parser.parse_args()
    
    explain_prediction(args.p1, args.p2)
