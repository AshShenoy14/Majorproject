from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
import joblib
import os
from src.utils.calibration import PlattScaler
from typing import Tuple

META_FEATURE_NAMES = ["p_seq", "p_graph", "conf_seq", "conf_graph", "diff", "max_conf", "consensus"]
N_META_FEATURES = len(META_FEATURE_NAMES)  # 7 - order must match train_ensemble.py


class PPIEnsemble:
    def __init__(self, meta_model_path: str = None):
        """
        Deep Stacking Ensemble using an XGBoost meta-learner.
        Meta-features (7): [p_seq, p_graph, conf_seq, conf_graph, diff, max_conf, consensus]
          conf_x = |p_x - 0.5|, diff = |p_seq - p_graph|, max_conf = max(conf_seq, conf_graph), consensus = p_seq * p_graph.
        p_graph is the CALIBRATED GraphSAGE probability. If <model dir>/graph_calibrator.json exists it is loaded
        and applied inside predict(), which therefore takes the RAW GraphSAGE probability.
        """
        self.meta_model = None
        self.graph_calibrator = None
        if meta_model_path:
            try:
                self.meta_model = joblib.load(meta_model_path)
                print(f"Loaded meta-learner from {meta_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load meta-learner from {meta_model_path}: {e}")
            n_in = getattr(self.meta_model, "n_features_in_", N_META_FEATURES)
            if n_in != N_META_FEATURES:
                raise RuntimeError(
                    f"Meta-learner at {meta_model_path} expects {n_in} features but the ensemble now uses "
                    f"{N_META_FEATURES} ({META_FEATURE_NAMES}). It is a stale model - retrain train_ensemble.py."
                )
            cal_path = os.path.join(os.path.dirname(str(meta_model_path)), "graph_calibrator.json")
            if os.path.exists(cal_path):
                self.graph_calibrator = PlattScaler.load(cal_path)
                print(f"Loaded GraphSAGE calibrator from {cal_path}")

    def calibrate_graph(self, graph_probs: np.ndarray) -> np.ndarray:
        """Raw GraphSAGE probabilities -> calibrated probabilities (identity if no calibrator is attached)."""
        graph_probs = np.asarray(graph_probs, dtype=np.float64)
        return graph_probs if self.graph_calibrator is None else self.graph_calibrator.transform_probs(graph_probs)

    @staticmethod
    def _build_features(base_preds_1: np.ndarray, base_preds_2: np.ndarray) -> np.ndarray:
        conf_1 = np.abs(base_preds_1 - 0.5)
        conf_2 = np.abs(base_preds_2 - 0.5)

        disagreement = np.abs(base_preds_1 - base_preds_2)
        max_conf = np.maximum(conf_1, conf_2)

        # Interaction feature: sequence and graph consensus
        consensus = base_preds_1 * base_preds_2

        return np.column_stack((base_preds_1, base_preds_2, conf_1, conf_2, disagreement, max_conf, consensus))

    def train_stacking(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, labels: np.ndarray):
        """
        Trains an optimized XGBoost meta-learner on out-of-fold base model predictions and training labels.
        base_preds_2 must already be the (out-of-fold) CALIBRATED graph probabilities.
        """
        X = self._build_features(base_preds_1, base_preds_2)
        
        print(f"Training Deep Ensemble on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # High-performance XGBoost configuration for meta-learning
        self.meta_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        
        # Fit on (X, labels) without label-dependent error sample weights
        self.meta_model.fit(X, labels)
        
        train_acc = self.meta_model.score(X, labels)
        print(f"Ensemble training complete. Training Accuracy: {train_acc*100:.2f}%")

    def predict(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, method: str = "stacking") -> np.ndarray:
        """base_preds_1: sequence probs; base_preds_2: RAW graph probs (calibrated here when a calibrator is attached)."""
        if method == "soft_voting":
            raise ValueError("soft_voting fallback is disabled in production safety mode. Use trained XGBoost meta-learner ('stacking').")
            
        elif method == "stacking":
            if self.meta_model is None:
                raise RuntimeError("Meta-learner (XGBoost) model is not loaded or trained. Call train_stacking() or provide a valid meta_model_path.")
            
            X = self._build_features(np.asarray(base_preds_1), self.calibrate_graph(base_preds_2))
            return self.meta_model.predict_proba(X)[:, 1]
        
        else:
            raise ValueError(f"Unknown prediction method '{method}'. Supported method: 'stacking'.")

    def save(self, path: str):
        if self.meta_model:
            joblib.dump(self.meta_model, path)
            if self.graph_calibrator is not None:
                self.graph_calibrator.save(os.path.join(os.path.dirname(str(path)), "graph_calibrator.json"))
            print(f"Deep Ensemble saved to {path}")
        else:
            raise RuntimeError("Cannot save un-trained ensemble meta-learner.")

