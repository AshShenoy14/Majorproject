import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.models.graph_model import GNNLinkPredictor
from pathlib import Path

def evaluate_best():
    device = torch.device('cpu')
    PROCESSED_DATA_DIR = Path(r'e:\majorproject\data\processed')
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    best_pth = Path(r'e:\majorproject\models\graph_model_best.pth')
    
    # Load data
    data = torch.load(graph_path, map_location=device, weights_only=False)
    node_mapping = torch.load(mapping_path, weights_only=False)
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    
    # Reconstruct Model
    model = GNNLinkPredictor(in_channels=data.x.shape[1], hidden_channels=128, heads=4).to(device)
    
    # Load Best Weights
    if best_pth.exists():
        model.load_state_dict(torch.load(best_pth, map_location=device, weights_only=False))
        print(f"Loaded BEST model from {best_pth}")
    else:
        print(f"ERROR: Best model not found at {best_pth}")
        return

    model.eval()
    
    # Logic from train_graph_model.py
    def get_edge_label_index(df):
        src, dst, labels = [], [], []
        for _, row in df.iterrows():
            p1, p2 = str(row["protein1"]), str(row["protein2"])
            if p1 in node_mapping and p2 in node_mapping:
                src.append(node_mapping[p1])
                dst.append(node_mapping[p2])
                labels.append(row["label"])
        return (torch.tensor([src, dst], dtype=torch.long),
                torch.tensor(labels, dtype=torch.float32))

    val_edges, val_labels = get_edge_label_index(val_df)

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        # Chunked decode for memory
        probs = []
        chunk_size = 500
        for i in range(0, val_edges.size(1), chunk_size):
            s = val_edges[0, i:i+chunk_size]
            d = val_edges[1, i:i+chunk_size]
            out = model.decode(z, s, d)
            probs.append(torch.sigmoid(out).cpu().numpy())
        
        all_probs = np.concatenate(probs)
        preds = (all_probs > 0.5).astype(float)
        
        y_true = val_labels.numpy()
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        
        print(f"\n--- EVALUATION RESULTS ---")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation F1-Score: {f1:.4f}")
        print(f"Total Predicted Interactions: {int(preds.sum())} / {len(preds)}")

if __name__ == "__main__":
    evaluate_best()
