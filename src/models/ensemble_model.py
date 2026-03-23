import xgboost as xgb
import numpy as np
import joblib
from typing import Tuple

class PPIEnsemble:
    def __init__(self, meta_model_path: str = None):
        """
        Ensemble model using Stacking with enhanced features.
        Features: [seq_prob, gat_prob, |seq_prob - 0.5|, |gat_prob - 0.5|]
        """
        self.meta_model = None
        if meta_model_path:
            try:
                self.meta_model = joblib.load(meta_model_path)
                print(f"Loaded meta-learner from {meta_model_path}")
            except:
                print("Could not load meta-learner. Path might be invalid or not exist yet.")

    @staticmethod
    def _build_features(base_preds_1: np.ndarray, base_preds_2: np.ndarray, bio_features: np.ndarray = None) -> np.ndarray:
        """
        Build enhanced feature matrix for the meta-learner.
        Features: [seq_prob, gat_prob, conf_seq, conf_gat, disagreement, max_conf, bio_features]
        """
        conf_1 = np.abs(base_preds_1 - 0.5)
        conf_2 = np.abs(base_preds_2 - 0.5)
        
        disagreement = np.abs(base_preds_1 - base_preds_2)
        max_conf = np.maximum(conf_1, conf_2)
        
        base_stack = np.column_stack((base_preds_1, base_preds_2, conf_1, conf_2, disagreement, max_conf))
        
        if bio_features is not None:
            if bio_features.ndim == 1:
                bio_features = bio_features.reshape(-1, 1)
            return np.hstack((base_stack, bio_features))
        
        return base_stack

    def train_stacking(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, labels: np.ndarray, bio_features: np.ndarray = None):
        """
        Trains the XGBoost meta-learner.
        args:
            base_preds_1: Predictions from Sequence Model
            base_preds_2: Predictions from Graph Model
            labels: True labels
            bio_features: Optional additional biological features (tabular)
        """
        X = self._build_features(base_preds_1, base_preds_2, bio_features)
        
        # Give higher weight to hard negatives: where seq says YES (>0.8) and label is NO 
        # and there's a significant gap (>0.2) with the Graph Model's prediction.
        weights = np.ones(len(labels))
        seq_graph_gap = base_preds_1 - base_preds_2
        hard_negatives = (base_preds_1 > 0.8) & (labels == 0) & (seq_graph_gap > 0.2)
        weights[hard_negatives] = 15.0  # Force XGBoost to care more about these specific traps!
        
        print(f"Training XGBoost Meta-Learner with {X.shape[1]} features and {np.sum(weights > 1)} high-weight samples...")
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        self.meta_model.fit(X, labels, sample_weight=weights)
        print("Meta-learner training complete.")

    def predict(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, bio_features: np.ndarray = None, method: str = "stacking") -> np.ndarray:
        """
        Predicts final probability.
        """
        if method == "soft_voting":
            return (base_preds_1 + base_preds_2) / 2.0
            
        elif method == "stacking":
            if self.meta_model is None:
                raise ValueError("Meta-learner not trained/loaded.")
            
            # Check model feature size to handle legacy 4-feature models
            n_expected = getattr(self.meta_model, "n_features_in_", 4)
            if n_expected == 4:
                bio_features = None
                
            X = self._build_features(base_preds_1, base_preds_2, bio_features)
            return self.meta_model.predict_proba(X)[:, 1]
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def save(self, path: str):
        if self.meta_model:
            joblib.dump(self.meta_model, path)
            print(f"Meta-learner saved to {path}")
