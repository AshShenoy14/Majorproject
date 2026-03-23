import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from typing import Tuple


class PPIEnsemble:
    """
    Stacking ensemble combining ESM-MLP and GAT predictions
    using a tuned XGBoost meta-learner.

    Features: [seq_prob, gat_prob, |seq_prob - 0.5|, |gat_prob - 0.5|]
    """

    def __init__(self, meta_model_path: str = None):
        self.meta_model = None
        if meta_model_path:
            try:
                self.meta_model = joblib.load(meta_model_path)
                print(f"Loaded meta-learner from {meta_model_path}")
            except Exception:
                print("Could not load meta-learner. Will need retraining.")

    @staticmethod
    def _build_features(base_preds_1: np.ndarray, base_preds_2: np.ndarray) -> np.ndarray:
        """
        Build enhanced 5-feature matrix for meta-learner:
        [seq_prob, gat_prob, seq_confidence, gat_confidence, model_disagreement]
        """
        conf_1 = np.abs(base_preds_1 - 0.5)
        conf_2 = np.abs(base_preds_2 - 0.5)
        disagreement = np.abs(base_preds_1 - base_preds_2)
        return np.column_stack([base_preds_1, base_preds_2, conf_1, conf_2, disagreement])

    def train_stacking(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray,
                       labels: np.ndarray):
        """
        Train XGBoost meta-learner with tuned hyperparameters.

        Args:
            base_preds_1: Sequence model probabilities (N,)
            base_preds_2: Graph model probabilities (N,)
            labels: True labels (N,)
        """
        X = self._build_features(base_preds_1, base_preds_2)

        # Check feature variance — warn if base models collapsed
        for i, name in enumerate(["SeqModel", "GraphModel"]):
            var = np.var(X[:, i])
            if var < 1e-4:
                print(f"  WARNING: {name} predictions have near-zero variance ({var:.6f})")
                print(f"    → Ensemble will rely heavily on the other model.")

        print(f"Training XGBoost Meta-Learner with {X.shape[1]} features...")

        # Compute scale_pos_weight for class imbalance
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  Class balance: {n_pos} pos / {n_neg} neg → scale_pos_weight={spw:.3f}")

        self.meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
        )
        self.meta_model.fit(X, labels)

        # Quick CV sanity check
        cv_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective='binary:logistic', eval_metric='logloss',
            use_label_encoder=False, random_state=42,
        )
        cv_scores = cross_val_score(cv_model, X, labels, cv=3, scoring="roc_auc")
        print(f"  Ensemble 3-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("Meta-learner training complete.")

    def predict(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray,
                method: str = "stacking") -> np.ndarray:
        """Predict interaction probability."""
        if method == "soft_voting":
            return (base_preds_1 + base_preds_2) / 2.0

        elif method == "stacking":
            if self.meta_model is None:
                raise ValueError("Meta-learner not trained/loaded.")
            X = self._build_features(base_preds_1, base_preds_2)
            return self.meta_model.predict_proba(X)[:, 1]

        else:
            raise ValueError(f"Unknown method: {method}")

    def save(self, path: str):
        """Save XGBoost meta-learner."""
        if self.meta_model:
            joblib.dump(self.meta_model, path)
            print(f"Meta-learner saved to {path}")

    @staticmethod
    def load(path: str):
        """Load saved XGBoost model."""
        return joblib.load(path)