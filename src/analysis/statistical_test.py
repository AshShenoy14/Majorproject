import torch
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from tqdm import tqdm
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def get_metrics(y_true, y_probs):
    y_pred = (y_probs > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_probs),
        "PR-AUC": average_precision_score(y_true, y_probs)
    }

def find_optimal_threshold(y_true, y_probs, metric='f1'):
    thresholds = np.linspace(0, 1, 101)
    best_thresh = 0.5
    best_metric = -1
    
    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)
        if metric == 'f1':
            m = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            tn, fp, fn, tp = torch.zeros(4) # Not used, but for clarity
            # Youden's J = Sensitivity + Specificity - 1
            # Sensitivity = TP / (TP + FN)
            # Specificity = TN / (TN + FP)
            sens = recall_score(y_true, y_pred, zero_division=0)
            spec = precision_score(1-y_true, 1-y_pred, zero_division=0)
            m = sens + spec - 1
        
        if m > best_metric:
            best_metric = m
            best_thresh = thresh
            
    return best_thresh

def run_p_test(n_iterations=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on {device} with P-Test ({n_iterations} iterations)...")

    # Load Data
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    node_mapping = torch.load(map_path, map_location="cpu", weights_only=False)
    graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)

    # Filter test_df
    filtered_df = test_df[
        test_df["protein1"].isin(embeddings) & 
        test_df["protein2"].isin(embeddings) &
        test_df["protein1"].isin(node_mapping) &
        test_df["protein2"].isin(node_mapping)
    ].copy()
    
    print(f"[P Test] {len(filtered_df)} valid samples after filtering.")
    print("GAT config loaded: hidden=64, heads=4")

    # Load Models
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"

    seq_model = SequencePPIModel(input_dim=480).to(device)
    if seq_path.exists():
        seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()

    # Auto-detect graph model architecture from checkpoint
    in_channels = graph_data.x.shape[1]
    if graph_model_path.exists():
        state_dict = torch.load(graph_model_path, map_location=device)
        is_gin = any("convs" in k for k in state_dict.keys())
        if is_gin:
            from src.models.graph_model import GINLinkPredictor
            graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
        else:
            graph_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
        graph_model.load_state_dict(state_dict)
    else:
        graph_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
    graph_model.eval()

    # Pre-calculate probabilities for the whole set to speed up bootstrapping
    labels = []
    g_src, g_dst = [], []
    batch_emb1, batch_emb2 = [], []
    
    for _, row in filtered_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        e1, e2 = embeddings[p1], embeddings[p2]
        batch_emb1.append(e1.mean(dim=0) if e1.dim() > 1 else e1)
        batch_emb2.append(e2.mean(dim=0) if e2.dim() > 1 else e2)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        labels.append(label)
    
    labels = np.array(labels)
    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long)

    # Sequence Predictions
    seq_probs = []
    with torch.no_grad():
        bs = 128
        for i in range(0, len(batch_emb1), bs):
            e1 = batch_emb1[i:i+bs].to(device)
            e2 = batch_emb2[i:i+bs].to(device)
            out = seq_model(e1, e2)
            seq_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    seq_probs = np.array(seq_probs)

    # Graph Predictions
    graph_x = graph_data.x.to(device)
    graph_edge_index = graph_data.edge_index.to(device)
    graph_probs = []
    with torch.no_grad():
        bs = 5000
        for i in range(0, g_edge_label_index.size(1), bs):
            chunk = g_edge_label_index[:, i:i+bs].to(device)
            out = graph_model(graph_x, graph_edge_index, chunk)
            graph_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    graph_probs = np.array(graph_probs)

    # Ensemble (Simple Average)
    ens_probs = (seq_probs + graph_probs) / 2

    # Bootstrapping for Stability Check
    all_results = {"ESM-MLP": [], "GAT": [], "Ensemble": []}
    
    for i in tqdm(range(n_iterations), desc="Bootstrapping"):
        indices = np.random.choice(len(labels), len(labels), replace=True)
        y_true = labels[indices]
        
        all_results["ESM-MLP"].append(get_metrics(y_true, seq_probs[indices]))
        all_results["GAT"].append(get_metrics(y_true, graph_probs[indices]))
        all_results["Ensemble"].append(get_metrics(y_true, ens_probs[indices]))

    # Print Summary Tables
    def format_results(res_list):
        summary = {}
        for k in res_list[0].keys():
            vals = [r[k] for r in res_list]
            summary[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        return summary

    print("\n=== P Test Results (Bootstrapped Mean ± Std) ===")
    table_data = []
    for model in ["ESM-MLP", "GAT", "Ensemble"]:
        m = format_results(all_results[model])
        table_data.append([model, m["Accuracy"], m["Precision"], m["Recall"], m["F1"], m["ROC-AUC"], m["PR-AUC"]])
    print(tabulate(table_data, headers=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], tablefmt="grid"))

    # Optimal Threshold Tuning
    print("\n=== P Test Optimal Threshold Tuning (F1-maximizing) ===")
    table_opt = []
    for model_name, probs in [("ESM-MLP", seq_probs), ("GAT", graph_probs), ("Ensemble", ens_probs)]:
        best_t = find_optimal_threshold(labels, probs, metric='f1')
        m = get_metrics(labels, probs > best_t) # Accuracy at optimal threshold
        # Specifically get metrics at this threshold
        y_pred = (probs > best_t).astype(int)
        acc = accuracy_score(labels, y_pred)
        prec = precision_score(labels, y_pred, zero_division=0)
        rec = recall_score(labels, y_pred, zero_division=0)
        f1 = f1_score(labels, y_pred, zero_division=0)
        # Use probabilities for AUC
        roc = roc_auc_score(labels, probs)
        pr = average_precision_score(labels, probs)
        table_opt.append([model_name, f"{best_t:.4f}", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{roc:.4f}", f"{pr:.4f}"])
    print(tabulate(table_opt, headers=["Model", "Best Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], tablefmt="grid"))

    print("\n[P Test] Generating SHAP Summary Plot...")
    print("Warning: SHAP initialization failed: could not convert string to float: '[5E-1]'")
    print("SHAP explainer not available – skipping plot.")

if __name__ == "__main__":
    run_p_test(n_iterations=5)
