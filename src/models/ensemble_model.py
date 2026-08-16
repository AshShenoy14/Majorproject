from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
import joblib
from typing import Tuple

class PPIEnsemble:
    def __init__(self, meta_model_path: str = None):
        """
        Deep Stacking Ensemble using XGBoost meta-learner.
        Features: [seq_prob, gat_prob, conf_seq, conf_gat, disagreement, max_conf, consensus, (bio_features)]
        """
        self.meta_model = None
        if meta_model_path:
            try:
                self.meta_model = joblib.load(meta_model_path)
                print(f"Loaded meta-learner from {meta_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load meta-learner from {meta_model_path}: {e}")

    @staticmethod
    def _build_features(base_preds_1: np.ndarray, base_preds_2: np.ndarray, bio_features: np.ndarray = None) -> np.ndarray:
        conf_1 = np.abs(base_preds_1 - 0.5)
        conf_2 = np.abs(base_preds_2 - 0.5)
        
        disagreement = np.abs(base_preds_1 - base_preds_2)
        max_conf = np.maximum(conf_1, conf_2)
        
        # Interaction feature: sequence and graph consensus
        consensus = (base_preds_1 * base_preds_2)
        
        base_stack = np.column_stack((
            base_preds_1, 
            base_preds_2, 
            conf_1, 
            conf_2, 
            disagreement, 
            max_conf,
            consensus
        ))
        
        if bio_features is not None:
            if bio_features.ndim == 1:
                bio_features = bio_features.reshape(-1, 1)
            return np.hstack((base_stack, bio_features))
        
        return base_stack

    def train_stacking(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, labels: np.ndarray, bio_features: np.ndarray = None):
        """
        Trains an optimized XGBoost meta-learner on out-of-fold base model predictions and training labels.
        """
        X = self._build_features(base_preds_1, base_preds_2, bio_features)
        
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

    def predict(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, bio_features: np.ndarray = None, method: str = "stacking") -> np.ndarray:
        if method == "soft_voting":
            raise ValueError("soft_voting fallback is disabled in production safety mode. Use trained XGBoost meta-learner ('stacking').")
            
        elif method == "stacking":
            if self.meta_model is None:
                raise RuntimeError("Meta-learner (XGBoost) model is not loaded or trained. Call train_stacking() or provide a valid meta_model_path.")
            
            X = self._build_features(base_preds_1, base_preds_2, bio_features)
            return self.meta_model.predict_proba(X)[:, 1]
        
        else:
            raise ValueError(f"Unknown prediction method '{method}'. Supported method: 'stacking'.")

    def save(self, path: str):
        if self.meta_model:
            joblib.dump(self.meta_model, path)
            print(f"Deep Ensemble saved to {path}")
        else:
            raise RuntimeError("Cannot save un-trained ensemble meta-learner.")

