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

    def explain_prediction(self, seq_prob: float, graph_prob: float, conf_seq: float, conf_graph: float, bio_match: float):
        """
        Explains a single prediction made by the ensemble using the 5 features.
        Features: [seq, graph, conf_seq, conf_graph, bio_match]
        """
        if self.explainer is None:
            return np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
        # Matrix matching the features the meta-learner was trained on
        X = np.array([[seq_prob, graph_prob, conf_seq, conf_graph, bio_match]])
        
        # Handle legacy 4-feature models
        n_expected = getattr(self.model, "n_features_in_", 4)
        if n_expected == 4:
            X = X[:, :4]
            
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
             if len(shap_values) == 2:
                  return shap_values[1]
             return shap_values[0]
             
        return shap_values

    def save_summary_plot(self, X: np.ndarray, feature_names=["ESM-MLP", "GAT", "|ESM-0.5|", "|GAT-0.5|", "Bio Localization"], title="SHAP Summary Plot", output_path="shap_summary.png"):
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

def plot_gat_attention(attention_weights, edge_index):
    """
    Placeholder for plotting GAT attention weights on the graph.
    Requires extraction of attention weights from GAT model.
    """
    pass
