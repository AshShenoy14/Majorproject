import torch
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def evaluate_subset(n_samples=100):
    device = torch.device("cpu")
    print(f"Evaluating subset of {n_samples} samples on {device}...")

    # Load test data
    test_path = PROCESSED_DATA_DIR / "test.csv"
    test_df = pd.read_csv(test_path).head(n_samples)

    # Load mappings
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    
    node_mapping = torch.load(map_path, map_location="cpu", weights_only=False)
    graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)

    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=64).to(device)
    if graph_model_path.exists():
         graph_model.load_state_dict(torch.load(graph_model_path, map_location=device))
    graph_model.eval()

    g_src, g_dst, labels = [], [], []
    for _, row in test_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        if p1 in node_mapping and p2 in node_mapping:
            g_src.append(node_mapping[p1])
            g_dst.append(node_mapping[p2])
            labels.append(label)

    labels = np.array(labels)
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long)
    
    graph_x = graph_data.x.to(device)
    graph_edge_index = graph_data.edge_index.to(device)
    
    with torch.no_grad():
        out = graph_model(graph_x, graph_edge_index, g_edge_label_index)
        graph_probs = torch.sigmoid(out).numpy().flatten()
    
    y_pred = (graph_probs > 0.5).astype(int)
    
    print("\n=== Current GAT Metrics (Subset) ===")
    print(f"Accuracy:  {accuracy_score(labels, y_pred):.4f}")
    print(f"Precision: {precision_score(labels, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(labels, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(labels, graph_probs):.4f}")
    print(f"PR-AUC:    {average_precision_score(labels, graph_probs):.4f}")

if __name__ == "__main__":
    evaluate_subset()
