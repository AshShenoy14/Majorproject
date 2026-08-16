import torch
import pandas as pd
import numpy as np
import joblib
import os
import sys
import argparse
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
    Sweep thresholds from 0.1 to 0.9 and find the optimal one on validation data.
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
    plt.title(f'Confusion Matrix: {title}\n(Validation-Selected Threshold={threshold:.2f})')
    
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

def calc_metrics(y_true, y_prob, threshold=0.5):
    """Calculates evaluation metrics at a fixed threshold."""
    y_pred = (y_prob > threshold).astype(int)
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        roc_auc_score(y_true, y_prob),
        average_precision_score(y_true, y_prob)
    ]

def get_model_predictions(df, seq_model, graph_model, ensemble_model, embeddings, node_mapping, graph_data, device, desc="Inference"):
    """
    Generates sequence, graph, and ensemble predictions for a given dataset dataframe.
    Constructs the exact 8-feature matrix for the XGBoost ensemble.
    """
    filtered_df = df[
        df["protein1"].isin(embeddings) & 
        df["protein2"].isin(embeddings) &
        df["protein1"].isin(node_mapping) &
        df["protein2"].isin(node_mapping)
    ].copy().reset_index(drop=True)

    batch_emb1, batch_emb2, g_src, g_dst, labels = [], [], [], [], []
    for _, row in filtered_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        e1, e2 = embeddings[p1], embeddings[p2]
        
        e1_base = e1.mean(dim=0) if e1.dim() > 1 else e1
        e2_base = e2.mean(dim=0) if e2.dim() > 1 else e2
        
        batch_emb1.append(e1_base)
        batch_emb2.append(e2_base)
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
        z = graph_model.encode(graph_x, graph_edge_index)
        batch_size = 5000
        for i in tqdm(range(0, g_edge_label_index.size(1), batch_size), desc=f"{desc} (Graph)"):
            chunk = g_edge_label_index[:, i:i+batch_size].to(device)
            out = graph_model.decode(z, chunk[0], chunk[1])
            probs = torch.sigmoid(out)
            graph_preds.extend(probs.cpu().numpy().flatten())
    graph_preds = np.array(graph_preds)

    # Predict Ensemble — 8 features
    ens_preds = None
    X_8feat = None
    if ensemble_model:
        conf_seq = np.abs(seq_preds - 0.5)
        conf_gat = np.abs(graph_preds - 0.5)
        disagreement = np.abs(seq_preds - graph_preds)
        max_conf = np.maximum(conf_seq, conf_gat)
        consensus = seq_preds * graph_preds
        
        from src.analysis.biological_managers import BiologicalManager
        bio_manager = BiologicalManager()
        bio_scores = []
        for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=f"{desc} (Bio)"):
            p1, p2 = row["protein1"], row["protein2"]
            comp = bio_manager.check_localization_compatibility(p1, p2, fetch_missing=False)
            bio_scores.append(comp.get("score", 0.5))
        
        bio_scores_np = np.array(bio_scores)
        
        # 8-Feature Stack: [seq_prob, gat_prob, conf_seq, conf_gat, disagreement, max_conf, consensus, bio_score]
        X_8feat = np.column_stack((
            seq_preds, 
            graph_preds, 
            conf_seq, 
            conf_gat, 
            disagreement, 
            max_conf, 
            consensus,
            bio_scores_np
        ))
        
        ens_preds = ensemble_model.predict_proba(X_8feat)[:, 1]

    return labels, seq_preds, graph_preds, ens_preds, X_8feat, filtered_df

def evaluate_models(dry_run=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on {device}...")

    # Load dataset paths
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    if not val_path.exists() or not test_path.exists():
        print(f"Required dataset files (val.csv, test.csv) missing.")
        return

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if dry_run:
        print("[DRY-RUN] Subsampling val.csv and test.csv to 200 samples for verification...")
        val_df = val_df.iloc[:200].copy().reset_index(drop=True)
        test_df = test_df.iloc[:200].copy().reset_index(drop=True)

    print(f"Loaded {len(val_df)} validation samples and {len(test_df)} test samples.")

    # Load embeddings and graph metadata
    from src.utils.bio_encoder import BioFeatureEncoder
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0

    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    
    if not emb_path.exists() or not map_path.exists() or not graph_data_path.exists():
        print("Required processed data (embeddings, mapping, graph) missing.")
        return

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, map_location="cpu", weights_only=False)
    graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)

    # Load Checkpoints
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"

    sample_emb = next(iter(embeddings.values()))
    esm_dim = sample_emb.shape[-1]
    
    seq_model = SequencePPIModel(input_dim=esm_dim, hidden_dim=1024).to(device)
    if seq_path.exists():
        print(f"Loading Sequence Model checkpoint ({seq_path.name})...")
        seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()

    in_channels = graph_data.x.shape[1]
    if graph_model_path.exists():
        print(f"Loading Graph Model checkpoint ({graph_model_path.name})...")
        state_dict = torch.load(graph_model_path, map_location=device, weights_only=False)
        is_gin = any("convs" in k for k in state_dict.keys())
        if is_gin:
            graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
        else:
            graph_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
        graph_model.load_state_dict(state_dict)
    graph_model.eval()

    ensemble_model = joblib.load(ensemble_path) if ensemble_path.exists() else None
    if ensemble_model:
        print(f"Loaded XGBoost Meta-Learner checkpoint ({ensemble_path.name}).")

    # =========================================================================
    # STEP 1: VALIDATION THRESHOLD SELECTION (val.csv ONLY)
    # =========================================================================
    print("\n--- STEP 1: Selecting Decision Thresholds on Validation Set (val.csv ONLY) ---")
    val_labels, val_seq, val_graph, val_ens, val_X_8feat, _ = get_model_predictions(
        val_df, seq_model, graph_model, ensemble_model, embeddings, node_mapping, graph_data, device, desc="Val Inference"
    )

    val_thresh_seq, _ = find_optimal_threshold(val_labels, val_seq, method="f1")
    val_thresh_graph, _ = find_optimal_threshold(val_labels, val_graph, method="f1")
    val_thresh_ens = 0.5
    if val_ens is not None:
        val_thresh_ens, _ = find_optimal_threshold(val_labels, val_ens, method="f1")

    print("\n[VALIDATION THRESHOLDS SELECTED]")
    print(f"  Sequence Model Optimal Threshold (val.csv): {val_thresh_seq:.4f}")
    print(f"  Graph Model Optimal Threshold    (val.csv): {val_thresh_graph:.4f}")
    if val_ens is not None:
        print(f"  Ensemble Model Optimal Threshold (val.csv): {val_thresh_ens:.4f}")

    if dry_run:
        print("\n[VERIFIED] Threshold selection was performed strictly on val.csv labels.")
        print("[VERIFIED] test.csv has NOT been passed to find_optimal_threshold().")

    # =========================================================================
    # STEP 2: LEAKAGE-FREE FINAL TEST EVALUATION (test.csv)
    # =========================================================================
    print("\n--- STEP 2: Evaluating on Test Set (test.csv) using Validation Thresholds ---")
    test_labels, test_seq, test_graph, test_ens, test_X_8feat, _ = get_model_predictions(
        test_df, seq_model, graph_model, ensemble_model, embeddings, node_mapping, graph_data, device, desc="Test Inference"
    )

    # 1. Baseline Evaluation (Default threshold=0.5)
    std_results = []
    std_results.append(["Sequence-Only (ESM-MLP)"] + calc_metrics(test_labels, test_seq, threshold=0.5))
    std_results.append(["Graph-Only (GAT)"] + calc_metrics(test_labels, test_graph, threshold=0.5))
    if test_ens is not None:
        std_results.append(["Full Ensemble (XGBoost)"] + calc_metrics(test_labels, test_ens, threshold=0.5))

    headers = ["Component", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n=== Final Test Performance (Default Threshold = 0.5) ===")
    print(tabulate(std_results, headers=headers, floatfmt=".4f", tablefmt="grid"))

    # 2. Final Leakage-Free Test Evaluation (using Validation-Selected Thresholds)
    final_results = []
    final_results.append(["Sequence-Only (ESM-MLP)", val_thresh_seq] + calc_metrics(test_labels, test_seq, threshold=val_thresh_seq))
    final_results.append(["Graph-Only (GAT)", val_thresh_graph] + calc_metrics(test_labels, test_graph, threshold=val_thresh_graph))
    if test_ens is not None:
        final_results.append(["Full Ensemble (XGBoost)", val_thresh_ens] + calc_metrics(test_labels, test_ens, threshold=val_thresh_ens))

    final_headers = ["Component", "Val-Selected Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n=== FINAL LEAKAGE-FREE TEST RESULTS (Validation-Selected Thresholds) ===")
    print(tabulate(final_results, headers=final_headers, floatfmt=".4f", tablefmt="grid"))

    # 3. Generate Visualizations for Test Set using Validation-Selected Thresholds
    eval_dir = PROJECT_ROOT / "assets" / "evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    eval_configs = [
        ("Sequence-Only", test_seq, val_thresh_seq),
        ("Graph-Only", test_graph, val_thresh_graph)
    ]
    if test_ens is not None:
        eval_configs.append(("Ensemble", test_ens, val_thresh_ens))

    for name, preds, fixed_t in eval_configs:
        plot_confusion_matrix(
            test_labels, preds, threshold=fixed_t, 
            title=name, 
            save_path=eval_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        )
        plot_threshold_analysis(
            test_labels, preds, 
            title=name, 
            save_path=eval_dir / f"threshold_analysis_{name.lower().replace(' ', '_')}.png"
        )

    # =========================================================================
    # STEP 3: SHAP EXPLAINABILITY (8-FEATURE MATRIX VERIFICATION)
    # =========================================================================
    if ensemble_model and test_X_8feat is not None:
        print("\n--- SHAP Summary Plot Generation (8-Feature Matrix) ---")
        assert test_X_8feat.shape[1] == 8, f"ERROR: SHAP input matrix must have 8 features, but got {test_X_8feat.shape[1]}!"
        print(f"[VERIFIED] SHAP matrix dimension: {test_X_8feat.shape} (8 features).")
        
        try:
            explainer = PPIExplainer(str(ensemble_path))
            feature_names = [
                'seq_prob', 'gat_prob', 'conf_seq', 'conf_gat', 
                'disagreement', 'max_conf', 'consensus', 'bio_score'
            ]
            explainer.save_summary_plot(
                test_X_8feat, 
                output_path=str(PROJECT_ROOT / "data" / "processed" / "plots" / "shap_summary.png"),
                feature_names=feature_names
            )
        except Exception as e:
            print(f"SHAP summary plot generation note: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-Free Ensemble Evaluation Protocol")
    parser.add_argument("--dry_run", action="store_true", help="Run quick 200-sample verification pass without modifying models or running full test evaluation.")
    args = parser.parse_args()

    evaluate_models(dry_run=args.dry_run)
