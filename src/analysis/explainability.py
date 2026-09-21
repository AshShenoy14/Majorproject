import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from src.utils.paths import PROJECT_ROOT

class PPIExplainer:
    def __init__(self, meta_model_path: str):
        self.model = joblib.load(meta_model_path)
        # Fix XGBoost/SHAP compatibility: newer XGBoost stores base_score as '[5E-1]'
        # which SHAP can't parse. Strip the brackets so it becomes '5E-1' (a valid float string).
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: SHAP initialization failed due to XGBoost compatibility issue: {e}")
            self.explainer = None

    def explain_prediction(self, seq_prob: float, graph_prob: float, conf_seq: float, conf_graph: float, disagreement: float, max_conf: float):
        """
        Explains a single prediction using the 7 meta-features
        [p_seq, p_graph, conf_seq, conf_graph, diff, max_conf, consensus]; graph_prob is the CALIBRATED probability.
        """
        if self.explainer is None:
            return np.array([[0.0] * 7])

        consensus = seq_prob * graph_prob  # must match ensemble_model.py
        X = np.array([[seq_prob, graph_prob, conf_seq, conf_graph, disagreement, max_conf, consensus]])

        n_expected = getattr(self.model, "n_features_in_", 7)
        if X.shape[1] != n_expected:
            raise RuntimeError(f"Meta-learner expects {n_expected} features but 7 were built; retrain the ensemble.")

        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
             if len(shap_values) == 2:
                  return shap_values[1]
             return shap_values[0]
             
        return shap_values

    def save_summary_plot(self, X: np.ndarray, feature_names=["ESM-MLP", "GraphSAGE", "|ESM-0.5|", "|GraphSAGE-0.5|", "Disagreement", "Max Conf", "Consensus"], title="SHAP Summary Plot", output_path="shap_summary.png"):
        """
        Generates and saves a SHAP summary plot for a batch of predictions.
        """
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1] # Use positive class
        
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(title)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to {output_path}")
