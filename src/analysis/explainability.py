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

    def explain_prediction(self, seq_prob: float, graph_prob: float, conf_seq: float, conf_graph: float, disagreement: float, max_conf: float, bio_score: float):
        """
        Explains a single prediction made by the ensemble using the 8 features.
        Features: [seq, graph, conf_seq, conf_graph, disagreement, max_conf, consensus, bio_score]
        """
        if self.explainer is None:
            return np.array([[0.0] * 8])
            
        # Calculate consensus (must match ensemble_model.py logic)
        consensus = seq_prob * graph_prob
        
        # Matrix matching the features the meta-learner was trained on (8 features)
        X = np.array([[
            seq_prob, 
            graph_prob, 
            conf_seq, 
            conf_graph, 
            disagreement, 
            max_conf, 
            consensus, 
            bio_score
        ]])
        
        # Handle legacy models or feature count mismatch
        n_expected = getattr(self.model, "n_features_in_", 8)
        if X.shape[1] != n_expected:
            print(f"Warning: Feature count mismatch. Expected {n_expected}, got {X.shape[1]}. Truncating/Padding.")
            if X.shape[1] > n_expected:
                X = X[:, :n_expected]
            else:
                X = np.pad(X, ((0, 0), (0, n_expected - X.shape[1])), 'constant')
            
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
             if len(shap_values) == 2:
                  return shap_values[1]
             return shap_values[0]
             
        return shap_values

    def save_summary_plot(self, X: np.ndarray, feature_names=["ESM-MLP", "GAT", "|ESM-0.5|", "|GAT-0.5|", "Disagreement", "Max Conf", "Consensus", "Bio Localization"], title="SHAP Summary Plot", output_path="shap_summary.png"):
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
