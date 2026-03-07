import torch
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.analysis.explainability import PPIExplainer
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on {device}...")

    # Load test data
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        return

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} test samples.")

    # Load embeddings and mapping
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    
    if not emb_path.exists() or not map_path.exists() or not graph_data_path.exists():
        print("Required processed data (embeddings, mapping, graph) missing.")
        return
        
    # Load large data to CPU to avoid CUDA OOM — only small batches are moved to GPU at inference time
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    # Convert float16 embeddings to float32 for model compatibility
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, map_location="cpu", weights_only=False)
    graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)

    # Filter test_df to valid entries
    filtered_df = test_df[
        test_df["protein1"].isin(embeddings) & 
        test_df["protein2"].isin(embeddings) &
        test_df["protein1"].isin(node_mapping) &
        test_df["protein2"].isin(node_mapping)
    ].copy()

    # Load Models
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"

    sample_emb = next(iter(embeddings.values()))
    seq_model = SequencePPIModel(input_dim=sample_emb.shape[0]).to(device)
    if seq_path.exists():
         seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()

    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=64).to(device)
    if graph_model_path.exists():
         graph_model.load_state_dict(torch.load(graph_model_path, map_location=device))
    graph_model.eval()

    ensemble_model = None
    if ensemble_path.exists():
         ensemble_model = joblib.load(ensemble_path)

    # Prepare batch data
    batch_emb1 = []
    batch_emb2 = []
    g_src = []
    g_dst = []
    labels = []

    for _, row in filtered_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        batch_emb1.append(embeddings[p1])
        batch_emb2.append(embeddings[p2])
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        labels.append(label)

    labels = np.array(labels)

    # Predict Sequence
    # Keep stacked embeddings on CPU; batches will be moved to device during inference
    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)
    
    seq_preds = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(batch_emb1), batch_size):
            e1 = batch_emb1[i:i+batch_size].to(device)
            e2 = batch_emb2[i:i+batch_size].to(device)
            out = seq_model(e1, e2)
            seq_preds.extend(out.cpu().numpy().flatten())
    seq_preds = np.array(seq_preds)

    # Predict Graph
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long)  # stays on CPU until chunked
    # Move graph node features and edges to device once (they are small compared to embeddings)
    graph_x = graph_data.x.to(device)
    graph_edge_index = graph_data.edge_index.to(device)
    graph_preds = []
    with torch.no_grad():
        batch_size = 10000
        for i in range(0, g_edge_label_index.size(1), batch_size):
            chunk = g_edge_label_index[:, i:i+batch_size].to(device)
            out = graph_model(graph_x, graph_edge_index, chunk)
            graph_preds.extend(out.cpu().numpy().flatten())
    graph_preds = np.array(graph_preds)

    # Predict Ensemble
    ens_preds = None
    if ensemble_model:
         X = np.column_stack((seq_preds, graph_preds))
         ens_preds = ensemble_model.predict_proba(X)[:, 1]

    # Calculate Metrics
    def calc_metrics(y_true, y_prob):
        y_pred = (y_prob > 0.5).astype(int)
        return [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
            roc_auc_score(y_true, y_prob),
            average_precision_score(y_true, y_prob)
        ]

    results = []
    results.append(["ESM-MLP"] + calc_metrics(labels, seq_preds))
    results.append(["GAT"] + calc_metrics(labels, graph_preds))
    if ens_preds is not None:
         results.append(["Ensemble"] + calc_metrics(labels, ens_preds))

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n" + tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))

    # Generate SHAP summary plot
    if ensemble_model:
        print("\nGenerating SHAP Summary Plot...")
        try:
            explainer = PPIExplainer(str(ensemble_path))
            X_shap = np.column_stack((seq_preds, graph_preds))
            explainer.save_summary_plot(X_shap, output_path=str(PROJECT_ROOT / "data" / "processed" / "plots" / "shap_summary.png"))
        except Exception as e:
            print(f"SHAP generation failed (XGBoost/SHAP version mismatch): {e}")
            print("Skipping SHAP plot. Metrics above are still valid.")

if __name__ == "__main__":
    evaluate_models()
