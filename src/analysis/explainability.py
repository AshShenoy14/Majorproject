import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import os
from src.utils.paths import PROJECT_ROOT


class PPIExplainer:
    def __init__(self, meta_model_path: str):
        self.model = joblib.load(meta_model_path)
        self.explainer = None

        try:
            # Fix XGBoost/SHAP compatibility:
            # Newer XGBoost stores base_score as '[5E-1]' with brackets
            # SHAP can't parse this as float. Strip brackets before init.
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                raw_config = booster.save_config()
                config = json.loads(raw_config)
                base_score = config["learner"]["learner_model_param"]["base_score"]
                if isinstance(base_score, str) and base_score.startswith("["):
                    fixed = base_score.strip("[]")
                    config["learner"]["learner_model_param"]["base_score"] = fixed
                    booster.load_config(json.dumps(config))
                    print(f"  Fixed XGBoost base_score: '{base_score}' -> '{fixed}'")

            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: SHAP initialization failed: {e}")
            self.explainer = None

    def explain_prediction(self, seq_prob: float, graph_prob: float,
                           conf_seq: float, conf_graph: float):
        """Explains a single prediction using SHAP values."""
        if self.explainer is None:
            return np.array([[0.0, 0.0, 0.0, 0.0]])

        X = np.array([[seq_prob, graph_prob, conf_seq, conf_graph]])
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            return shap_values[1] if len(shap_values) == 2 else shap_values[0]
        return shap_values

    def save_summary_plot(self, X: np.ndarray,
                          feature_names=None,
                          title="SHAP Summary Plot (P Test)",
                          output_path="shap_summary.png"):
        """Generates and saves a SHAP summary plot for a batch."""
        if self.explainer is None:
            print("SHAP explainer not available — skipping plot.")
            return

        if feature_names is None:
            feature_names = ["ESM-MLP", "GAT", "|ESM-0.5|", "|GAT-0.5|"]

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(title)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved to {output_path}")


def plot_gat_attention(attention_weights, edge_index):
    """Placeholder for GAT attention visualization."""
    pass