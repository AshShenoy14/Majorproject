import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
from src.utils.paths import PROJECT_ROOT

class PPIExplainer:
    def __init__(self, meta_model_path: str):
        self.model = joblib.load(meta_model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, seq_prob: float, graph_prob: float):
        """
        Explains a single prediction made by the ensemble.
        """
        # SHAP expects a matrix
        X = np.array([[seq_prob, graph_prob]])
        shap_values = self.explainer.shap_values(X)
        
        # Plotting
        # shap.force_plot(...)
        # For now, return values
        # SHAP returns (1, n_features)
        # Check instance type to handle different SHAP versions or Model types
        if isinstance(shap_values, list):
             # For some classifiers SHAP returns list of arrays (one per class)
             # We want positive class (index 1) usually, but XGBoost binary might return just one matrix
             if len(shap_values) == 2:
                  return shap_values[1]
             return shap_values[0]
             
        return shap_values

def plot_gat_attention(attention_weights, edge_index):
    """
    Placeholder for plotting GAT attention weights on the graph.
    Requires extraction of attention weights from GAT model.
    """
    pass
