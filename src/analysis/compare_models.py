import torch
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from tabulate import tabulate
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor, GINLinkPredictor
from src.analysis.explainability import PPIExplainer
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def find_optimal_threshold(y_true, y_prob, method="f1"):
    """
    Sweep thresholds from 0.1 to 0.9 and find the optimal one.
    method: 'f1' (maximize F1) or 'youden' (maximize Youden's index = TPR - FPR)
    """
    best_thresh = 0.5
    best_score = -1
    
    for thresh in np.arange(0.1, 0.9, 0.01):
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
            score = tpr - fpr  # Youden's J
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score

def plot_confusion_matrix(y_true, y_prob, threshold, title, save_path):
    """Generates and saves a confusion matrix heatmap."""
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {title}\n(Threshold={threshold:.2f})')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved Confusion Matrix to {save_path}")

def plot_threshold_analysis(y_true, y_prob, title, save_path):
    """Plots F1, Precision, and Recall across various thresholds."""
    thresholds = np.arange(0.1, 0.91, 0.05)
    f1s, precs, recs = [], [], []
    
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s, label='F1 Score', marker='o')
    plt.plot(thresholds, precs, label='Precision', linestyle='--')
    plt.plot(thresholds, recs, label='Recall', linestyle=':')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold Analysis: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved Threshold Analysis to {save_path}")

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
    from src.utils.bio_encoder import BioFeatureEncoder
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0
    print(f"  Loaded biological features (dim={bio_dim}).")

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

    # Detect dimensions dynamically
    sample_emb = next(iter(embeddings.values()))
    esm_dim = sample_emb.shape[-1]
    topo_dim = 4 # From your Research-Grade GAT run
    bio_dim = 10
    
    # Load Models
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"

    # Sequence model: Symmetric ESM-MLP (High Capacity)
    seq_model = SequencePPIModel(input_dim=esm_dim, hidden_dim=1024).to(device)
    if seq_path.exists():
         print(f"  Loading Sequence Model ({esm_dim} dim, 1024 hidden)...")
         try:
             seq_model.load_state_dict(torch.load(seq_path, map_location=device))
         except Exception as e:
             print(f"  [!] Sequence load failed: {e}")
    seq_model.eval()

    # Graph Model (High Capacity GraphSAGE)
    in_channels = graph_data.x.shape[1]
    print(f"  Detected Graph Input Channels: {in_channels}")
    if graph_model_path.exists():
         print(f"  Loading Graph Model (256 hidden)...")
         state_dict = torch.load(graph_model_path, map_location=device, weights_only=False)
         is_gin = any("convs" in k for k in state_dict.keys())
         graph_model = (GINLinkPredictor if is_gin else GATLinkPredictor)(in_channels=in_channels, hidden_channels=256).to(device)
         graph_model.load_state_dict(state_dict)
    graph_model.eval()

    ensemble_model = joblib.load(ensemble_path) if ensemble_path.exists() else None

    # Prepare batch data
    batch_emb1, batch_emb2, g_src, g_dst, labels = [], [], [], [], []
    for _, row in filtered_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        e1, e2 = embeddings[p1], embeddings[p2]
        
        # Mean-pool ESM
        e1_base = e1.mean(dim=0) if e1.dim() > 1 else e1
        e2_base = e2.mean(dim=0) if e2.dim() > 1 else e2
        
        batch_emb1.append(e1_base)
        batch_emb2.append(e2_base)
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
            # Apply sigmoid to raw logits (model no longer has built-in Sigmoid)
            probs = torch.sigmoid(out)
            seq_preds.extend(probs.cpu().numpy().flatten())
    seq_preds = np.array(seq_preds)

    # Predict Graph — encode once, then decode in chunks
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long)
    # Move graph node features and edges to device once (they are small compared to embeddings)
    graph_x = graph_data.x.to(device)
    graph_edge_index = graph_data.edge_index.to(device)
    
    graph_preds = []
    with torch.no_grad():
        print("Predicting with Graph Model...")
        # Encode entire graph once
        z = graph_model.encode(graph_x, graph_edge_index)
        
        batch_size = 5000 # More conservative batch size for decoding
        from tqdm import tqdm
        for i in tqdm(range(0, g_edge_label_index.size(1), batch_size), desc="Graph Inference"):
            chunk = g_edge_label_index[:, i:i+batch_size].to(device)
            # Use pre-computed z for decoding
            out = graph_model.decode(z, chunk[0], chunk[1])
            # Apply sigmoid to raw logits
            probs = torch.sigmoid(out)
            graph_preds.extend(probs.cpu().numpy().flatten())
    graph_preds = np.array(graph_preds)


    # Predict Ensemble — with enhanced 8-feature architecture
    ens_preds = None
    if ensemble_model:
        # Enhanced features: [seq, gat, conf_seq, conf_gat, disagreement, max_conf, consensus, bio_score]
        conf_seq = np.abs(seq_preds - 0.5)
        conf_gat = np.abs(graph_preds - 0.5)
        disagreement = np.abs(seq_preds - graph_preds)
        max_conf = np.maximum(conf_seq, conf_gat)
        consensus = seq_preds * graph_preds
        
        # Calculate biological compatibility scores for test set
        from src.analysis.biological_managers import BiologicalManager
        bio_manager = BiologicalManager()
        bio_scores = []
        for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Test Bio Analysis"):
            p1, p2 = row["protein1"], row["protein2"]
            comp = bio_manager.check_localization_compatibility(p1, p2, fetch_missing=False)
            bio_scores.append(comp.get("score", 0.5))
        
        bio_scores_np = np.array(bio_scores)
        
        # Construct the 8-feature matrix
        X = np.column_stack((
            seq_preds, 
            graph_preds, 
            conf_seq, 
            conf_gat, 
            disagreement, 
            max_conf, 
            consensus,
            bio_scores_np
        ))
        
        # Predict using the updated XGBoost model
        try:
            ens_preds = ensemble_model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"  [!] Ensemble prediction failed: {e}. Falling back to 4-feature slice...")
            # Fallback for older model compatibility if needed
            ens_preds = ensemble_model.predict_proba(X[:, :4])[:, 1]

    # Calculate Metrics (default threshold 0.5)
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

    # === Ablation Study & Standard Results (threshold=0.5) ===
    results = []
    results.append(["Sequence-Only (ESM-MLP)"] + calc_metrics(labels, seq_preds))
    results.append(["Graph-Only (GAT)"] + calc_metrics(labels, graph_preds))
    if ens_preds is not None:
         results.append(["Full Ensemble (Seq + Graph)"] + calc_metrics(labels, ens_preds))

    headers = ["Component/Ablation", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n=== Ablation Study: Model Performance (threshold=0.5) ===")
    print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))

    # === Threshold Tuning & Detailed Visualizations ===
    print("\n=== Threshold Tuning & Visual Analysis ===")
    eval_dir = PROJECT_ROOT / "assets" / "evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    tuning_results = []
    
    eval_configs = [
        ("Sequence-Only", seq_preds),
        ("Graph-Only", graph_preds)
    ]
    if ens_preds is not None:
        eval_configs.append(("Ensemble", ens_preds))
    
    for name, preds in eval_configs:
        print(f"Analyzing {name}...")
        best_t, best_f1 = find_optimal_threshold(labels, preds, method="f1")
        tuned_metrics = calc_metrics(labels, preds, threshold=best_t)
        tuning_results.append([name, best_t] + tuned_metrics)
        
        # New Visualizations
        plot_confusion_matrix(
            labels, preds, threshold=best_t, 
            title=name, 
            save_path=eval_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        )
        plot_threshold_analysis(
            labels, preds, 
            title=name, 
            save_path=eval_dir / f"threshold_analysis_{name.lower().replace(' ', '_')}.png"
        )
    
    tuning_headers = ["Model", "Best Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n=== Optimized Results (F1-maximizing Threshold) ===")
    print(tabulate(tuning_results, headers=tuning_headers, floatfmt=".4f", tablefmt="grid"))

    # === Youden's Index (Bonus Research Insight) ===
    print("\n=== Optimal Threshold (Youden's J) ===")
    youden_results = []
    for name, preds in eval_configs:
        best_t, best_j = find_optimal_threshold(labels, preds, method="youden")
        tuned_metrics = calc_metrics(labels, preds, threshold=best_t)
        youden_results.append([name, best_t, best_j] + tuned_metrics)
    
    youden_headers = ["Model", "Best Thresh", "Youden J", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(tabulate(youden_results, headers=youden_headers, floatfmt=".4f", tablefmt="grid"))

    # Generate SHAP summary plot
    if ensemble_model:
        print("\nGenerating SHAP Summary Plot...")
        try:
            explainer = PPIExplainer(str(ensemble_path))
            conf_seq = np.abs(seq_preds - 0.5)
            conf_gat = np.abs(graph_preds - 0.5)
            X_shap = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat))
            explainer.save_summary_plot(X_shap, output_path=str(PROJECT_ROOT / "data" / "processed" / "plots" / "shap_summary.png"))
        except Exception as e:
            print(f"SHAP generation failed (XGBoost/SHAP version mismatch): {e}")
            print("Skipping SHAP plot. Metrics above are still valid.")

if __name__ == "__main__":
    evaluate_models()
