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

EVAL_LABEL = "P Test"

def find_optimal_threshold(y_true, y_prob, method="f1"):
    best_thresh = 0.5
    best_score = -1

    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob > thresh).astype(int)
        if method == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif method == "youden":
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            score = tpr - fpr

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{EVAL_LABEL}] Evaluating models on {device}...")

    # Load test data
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"[{EVAL_LABEL}] Test data not found at {test_path}")
        return

    test_df = pd.read_csv(test_path)
    print(f"[{EVAL_LABEL}] Loaded {len(test_df)} test samples.")

    # Load embeddings and mapping
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    if not emb_path.exists() or not map_path.exists() or not graph_data_path.exists():
        print(f"[{EVAL_LABEL}] Required processed data (embeddings, mapping, graph) missing.")
        return

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, map_location="cpu", weights_only=False)
    graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)

    filtered_df = test_df[
        test_df["protein1"].isin(embeddings) &
        test_df["protein2"].isin(embeddings) &
        test_df["protein1"].isin(node_mapping) &
        test_df["protein2"].isin(node_mapping)
    ].copy()

    print(f"[{EVAL_LABEL}] {len(filtered_df)} valid samples after filtering.")

    # Load Models
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"

    input_dim = 320
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    if seq_path.exists():
        seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()

        # Load GAT config (auto-saved during training)
    gat_config_path = PROJECT_ROOT / "models" / "graph_model_config.pt"
    if gat_config_path.exists():
        gat_config = torch.load(gat_config_path, map_location="cpu", weights_only=False)
        gat_hidden = gat_config["hidden_channels"]
        gat_heads = gat_config.get("heads", 4)
        print(f"GAT config loaded: hidden={gat_hidden}, heads={gat_heads}")
    else:
        gat_hidden = 64
        gat_heads = 4
        print("GAT config not found — using defaults: hidden=64, heads=4")

    graph_model = GATLinkPredictor(
        in_channels=graph_data.x.shape[1],
        hidden_channels=gat_hidden,
        heads=gat_heads,
    ).to(device)
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
        e1 = embeddings[p1]
        e2 = embeddings[p2]
        batch_emb1.append(e1.mean(dim=0) if e1.dim() > 1 else e1)
        batch_emb2.append(e2.mean(dim=0) if e2.dim() > 1 else e2)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        labels.append(label)

    labels = np.array(labels)
    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)

    # Predict Sequence
    seq_preds = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(batch_emb1), batch_size):
            e1 = batch_emb1[i:i+batch_size].to(device)
            e2 = batch_emb2[i:i+batch_size].to(device)
            out = seq_model(e1, e2)
            probs = torch.sigmoid(out)
            seq_preds.extend(probs.cpu().numpy().flatten())
    seq_preds = np.array(seq_preds)

    # Predict Graph
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long)
    graph_x = graph_data.x.to(device)
    graph_edge_index = graph_data.edge_index.to(device)
    graph_preds = []
    with torch.no_grad():
        batch_size = 10000
        for i in range(0, g_edge_label_index.size(1), batch_size):
            chunk = g_edge_label_index[:, i:i+batch_size].to(device)
            out = graph_model(graph_x, graph_edge_index, chunk)
            probs = torch.sigmoid(out)
            graph_preds.extend(probs.cpu().numpy().flatten())
    graph_preds = np.array(graph_preds)

    # Predict Ensemble
    ens_preds = None
    if ensemble_model:
        conf_seq = np.abs(seq_preds - 0.5)
        conf_gat = np.abs(graph_preds - 0.5)
        X = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat))
        ens_preds = ensemble_model.predict_proba(X)[:, 1]

    # Metrics helper
    def calc_metrics(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob > threshold).astype(int)
        return [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
            roc_auc_score(y_true, y_prob),
            average_precision_score(y_true, y_prob)
        ]

    # === P Test Results (threshold=0.5) ===
    results = []
    results.append(["ESM-MLP"] + calc_metrics(labels, seq_preds))
    results.append(["GAT"] + calc_metrics(labels, graph_preds))
    if ens_preds is not None:
        results.append(["Ensemble"] + calc_metrics(labels, ens_preds))

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(f"\n=== {EVAL_LABEL} Results (threshold=0.5) ===")
    print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))

    # === P Test Optimal Threshold Tuning (F1) ===
    print(f"\n=== {EVAL_LABEL} Optimal Threshold Tuning (F1-maximizing) ===")
    tuning_results = []

    for name, preds in [("ESM-MLP", seq_preds), ("GAT", graph_preds)]:
        best_t, best_f1 = find_optimal_threshold(labels, preds, method="f1")
        tuned_metrics = calc_metrics(labels, preds, threshold=best_t)
        tuning_results.append([name, best_t] + tuned_metrics)

    if ens_preds is not None:
        best_t, best_f1 = find_optimal_threshold(labels, ens_preds, method="f1")
        tuned_metrics = calc_metrics(labels, ens_preds, threshold=best_t)
        tuning_results.append(["Ensemble", best_t] + tuned_metrics)

    tuning_headers = ["Model", "Best Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(tabulate(tuning_results, headers=tuning_headers, floatfmt=".4f", tablefmt="grid"))

    # === P Test Youden's Index ===
    print(f"\n=== {EVAL_LABEL} Optimal Threshold (Youden's J) ===")
    youden_results = []
    for name, preds in [("ESM-MLP", seq_preds), ("GAT", graph_preds)]:
        best_t, best_j = find_optimal_threshold(labels, preds, method="youden")
        tuned_metrics = calc_metrics(labels, preds, threshold=best_t)
        youden_results.append([name, best_t, best_j] + tuned_metrics)

    if ens_preds is not None:
        best_t, best_j = find_optimal_threshold(labels, ens_preds, method="youden")
        tuned_metrics = calc_metrics(labels, ens_preds, threshold=best_t)
        youden_results.append(["Ensemble", best_t, best_j] + tuned_metrics)

    youden_headers = ["Model", "Best Thresh", "Youden J", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(tabulate(youden_results, headers=youden_headers, floatfmt=".4f", tablefmt="grid"))

    # SHAP
    if ensemble_model:
        print(f"\n[{EVAL_LABEL}] Generating SHAP Summary Plot...")
        try:
            explainer = PPIExplainer(str(ensemble_path))
            conf_seq = np.abs(seq_preds - 0.5)
            conf_gat = np.abs(graph_preds - 0.5)
            X_shap = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat))
            explainer.save_summary_plot(
                X_shap,
                title=f"SHAP Summary Plot ({EVAL_LABEL})",
                output_path=str(PROJECT_ROOT / "data" / "processed" / "plots" / "shap_summary.png")
            )
        except Exception as e:
            print(f"SHAP generation failed: {e}")
            print("Skipping SHAP plot. Metrics above are still valid.")


if __name__ == "__main__":
    evaluate_models()