import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, List, Tuple

class MetricsReporter:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        y_preds = (y_probs >= threshold).astype(int)
        return {
            "Accuracy": accuracy_score(y_true, y_preds),
            "Precision": precision_score(y_true, y_preds, zero_division=0),
            "Recall": recall_score(y_true, y_preds, zero_division=0),
            "F1": f1_score(y_true, y_preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_probs),
            "PR-AUC": average_precision_score(y_true, y_probs)
        }

    @staticmethod
    def find_best_f1_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = -1
        best_thresh = 0.5
        best_metrics = {}

        for t in thresholds:
            metrics = MetricsReporter.calculate_metrics(y_true, y_probs, t)
            if metrics["F1"] > best_f1:
                best_f1 = metrics["F1"]
                best_thresh = t
                best_metrics = metrics
        
        return best_thresh, best_metrics

    @staticmethod
    def find_best_youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        thresholds = np.linspace(0.1, 0.9, 81)
        best_j = -1
        best_thresh = 0.5
        best_metrics = {}

        for t in thresholds:
            y_preds = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden_j = sensitivity + specificity - 1
            
            if youden_j > best_j:
                best_j = youden_j
                best_thresh = t
                best_metrics = MetricsReporter.calculate_metrics(y_true, y_probs, t)
        
        return best_thresh, best_j, best_metrics

    @staticmethod
    def print_table(title: str, results: List[Dict], headers: List[str]):
        print(f"\n=== {title} ===")
        
        # Create header row
        header_str = " | ".join(f"{h:^11}" for h in headers)
        print(f"+{'-' * (len(header_str) + 2)}+")
        print(f"| {header_str} |")
        print(f"+{'=' * (len(header_str) + 2)}+")

        for res in results:
            row_vals = []
            for h in headers:
                val = res.get(h, "")
                if isinstance(val, (float, np.float32, np.float64)):
                    row_vals.append(f"{val:11.4f}")
                else:
                    row_vals.append(f"{str(val):^11}")
            print(f"| {' | '.join(row_vals)} |")
            print(f"+{'-' * (len(header_str) + 2)}+")

def report_all_metrics(model_names: List[str], y_trues: List[np.ndarray], y_probs_list: List[np.ndarray]):
    reporter = MetricsReporter()
    
    # 1. Base Results (threshold=0.5)
    base_results = []
    headers_base = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    for name, y_true, y_probs in zip(model_names, y_trues, y_probs_list):
        m = reporter.calculate_metrics(y_true, y_probs, 0.5)
        m["Model"] = name
        base_results.append(m)
    reporter.print_table("Results (threshold=0.5)", base_results, headers_base)

    # 2. Optimal Threshold Tuning (F1-maximizing)
    f1_results = []
    headers_f1 = ["Model", "Best Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    for name, y_true, y_probs in zip(model_names, y_trues, y_probs_list):
        thresh, m = reporter.find_best_f1_threshold(y_true, y_probs)
        m["Model"] = name
        m["Best Thresh"] = thresh
        f1_results.append(m)
    reporter.print_table("Optimal Threshold Tuning (F1-maximizing)", f1_results, headers_f1)

    # 3. Optimal Threshold (Youden's J)
    j_results = []
    headers_j = ["Model", "Best Thresh", "Youden J", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    for name, y_true, y_probs in zip(model_names, y_trues, y_probs_list):
        thresh, j_val, m = reporter.find_best_youden_threshold(y_true, y_probs)
        m["Model"] = name
        m["Best Thresh"] = thresh
        m["Youden J"] = j_val
        j_results.append(m)
    reporter.print_table("Optimal Threshold (Youden's J)", j_results, headers_j)
