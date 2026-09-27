"""
GAT-specific Stacking Ensemble using an independent XGBoost meta-learner.

Strictly isolated under ESMGAT/ to protect the frozen GraphSAGE implementation.
Enforces identical 7 meta-features, identical XGBoost hyperparameters, and
strict path safety assertions.
"""
from pathlib import Path
import os
import joblib
import numpy as np
import xgboost as xgb
from typing import Tuple, Optional

from src.utils.calibration import PlattScaler

GAT_META_FEATURE_NAMES = [
    "p_seq",
    "p_gat",
    "conf_seq",
    "conf_gat",
    "diff",
    "max_conf",
    "consensus"
]
N_GAT_META_FEATURES = len(GAT_META_FEATURE_NAMES)  # Exactly 7


def assert_safe_gat_write(target_path: Path, project_root: Optional[Path] = None):
    """
    Enforces strict path safety:
    Raises RuntimeError if any write operation targets models/, checkpoints/,
    or any path outside ESMGAT/.
    """
    resolved = Path(target_path).resolve()
    if project_root is None:
        # Infer project root from this file location: ESMGAT/models/gat_ensemble.py -> project root
        project_root = Path(__file__).resolve().parent.parent.parent

    root_resolved = Path(project_root).resolve()
    esmgat_resolved = (root_resolved / "ESMGAT").resolve()

    # Must be inside ESMGAT/
    if esmgat_resolved not in resolved.parents and resolved != esmgat_resolved:
        raise RuntimeError(
            f"CRITICAL PATH SAFETY VIOLATION: Attempted write outside ESMGAT/: {resolved}\n"
            f"All GAT artifacts must reside strictly inside {esmgat_resolved}"
        )

    # Must NOT be in frozen GraphSAGE directories
    forbidden_roots = [
        (root_resolved / "models").resolve(),
        (root_resolved / "checkpoints").resolve(),
    ]
    for fb in forbidden_roots:
        if resolved == fb or fb in resolved.parents:
            raise RuntimeError(
                f"CRITICAL PATH SAFETY VIOLATION: Attempted write into frozen GraphSAGE directory: {resolved}"
            )


class GATPPIEnsemble:
    """
    Independent GAT Stacking Ensemble with XGBoost meta-learner.
    Uses the exact 7 meta-features derived from sequence and calibrated GAT probabilities:
      1. p_seq
      2. p_gat
      3. conf_seq = |p_seq - 0.5|
      4. conf_gat = |p_gat - 0.5|
      5. diff = |p_seq - p_gat|
      6. max_conf = max(conf_seq, conf_gat)
      7. consensus = p_seq * p_gat
    """
    def __init__(self, meta_model_path: Optional[str] = None, calibrator_path: Optional[str] = None):
        self.meta_model = None
        self.gat_calibrator = None

        if meta_model_path and os.path.exists(meta_model_path):
            self.load(meta_model_path, calibrator_path)

    @staticmethod
    def build_meta_features(p_seq: np.ndarray, p_gat: np.ndarray) -> np.ndarray:
        """
        Constructs the exact 7 meta-features:
        1. p_seq
        2. p_gat
        3. abs(p_seq - 0.5)
        4. abs(p_gat - 0.5)
        5. abs(p_seq - p_gat)
        6. max(conf_seq, conf_gat)
        7. p_seq * p_gat
        """
        p_seq = np.asarray(p_seq, dtype=np.float64)
        p_gat = np.asarray(p_gat, dtype=np.float64)

        if p_seq.shape != p_gat.shape:
            raise ValueError(f"Shape mismatch: p_seq {p_seq.shape} vs p_gat {p_gat.shape}")

        conf_seq = np.abs(p_seq - 0.5)
        conf_gat = np.abs(p_gat - 0.5)
        diff = np.abs(p_seq - p_gat)
        max_conf = np.maximum(conf_seq, conf_gat)
        consensus = p_seq * p_gat

        features = np.column_stack((p_seq, p_gat, conf_seq, conf_gat, diff, max_conf, consensus))
        assert features.shape[1] == N_GAT_META_FEATURES, f"Expected {N_GAT_META_FEATURES} features, got {features.shape[1]}"
        return features

    def calibrate_gat(self, gat_probs: np.ndarray) -> np.ndarray:
        """Raw GAT probabilities -> calibrated probabilities using Platt scaler."""
        gat_probs = np.asarray(gat_probs, dtype=np.float64)
        if self.gat_calibrator is None:
            return gat_probs
        return self.gat_calibrator.transform_probs(gat_probs)

    def train_stacking(
        self,
        oof_seq_preds: np.ndarray,
        oof_gat_cal_preds: np.ndarray,
        labels: np.ndarray,
        xgb_params: Optional[dict] = None
    ) -> xgb.XGBClassifier:
        """
        Trains an independent XGBoost meta-learner on out-of-fold base model predictions and training labels.
        oof_gat_cal_preds MUST already be the calibrated GAT probabilities.
        """
        X = self.build_meta_features(oof_seq_preds, oof_gat_cal_preds)
        y = np.asarray(labels, dtype=np.int32)

        print(f"[GATEnsemble] Training XGBoost meta-learner on {X.shape[0]} samples with {X.shape[1]} features...")

        # Exact hyperparameters matching GraphSAGE ensemble
        default_params = dict(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        if xgb_params:
            default_params.update(xgb_params)

        self.meta_model = xgb.XGBClassifier(**default_params)
        self.meta_model.fit(X, y)

        train_acc = self.meta_model.score(X, y)
        print(f"[GATEnsemble] Training complete. Training Accuracy: {train_acc * 100:.2f}%")
        return self.meta_model

    def predict_proba(self, p_seq: np.ndarray, p_gat_raw: np.ndarray) -> np.ndarray:
        """
        Inference with meta-learner:
        Accepts raw GAT predictions, calibrates them internally if calibrator attached,
        builds 7 meta-features, and predicts meta probabilities.
        """
        if self.meta_model is None:
            raise RuntimeError("GAT meta-learner model is not loaded or trained.")

        p_gat_cal = self.calibrate_gat(p_gat_raw)
        X = self.build_meta_features(p_seq, p_gat_cal)
        return self.meta_model.predict_proba(X)[:, 1]

    def save(self, model_path: str, calibrator_path: Optional[str] = None):
        """Saves XGBoost meta-learner with strict path safety assertions."""
        if self.meta_model is None:
            raise RuntimeError("Cannot save un-trained GAT meta-learner.")

        p = Path(model_path)
        assert_safe_gat_write(p)
        os.makedirs(p.parent, exist_ok=True)
        joblib.dump(self.meta_model, p)
        print(f"[GATEnsemble] Saved GAT XGBoost meta-learner to: {p}")

        if self.gat_calibrator is not None and calibrator_path is not None:
            cal_p = Path(calibrator_path)
            assert_safe_gat_write(cal_p)
            os.makedirs(cal_p.parent, exist_ok=True)
            self.gat_calibrator.save(str(cal_p))
            print(f"[GATEnsemble] Saved GAT calibrator to: {cal_p}")

    def load(self, model_path: str, calibrator_path: Optional[str] = None):
        """Loads GAT meta-learner and optional calibrator."""
        self.meta_model = joblib.load(model_path)
        n_in = getattr(self.meta_model, "n_features_in_", N_GAT_META_FEATURES)
        if n_in != N_GAT_META_FEATURES:
            raise RuntimeError(
                f"GAT Meta-learner at {model_path} expects {n_in} features but current pipeline uses {N_GAT_META_FEATURES}."
            )
        print(f"[GATEnsemble] Loaded GAT XGBoost meta-learner from: {model_path}")

        # Try loading calibrator if provided or if next to model
        if calibrator_path and os.path.exists(calibrator_path):
            self.gat_calibrator = PlattScaler.load(calibrator_path)
            print(f"[GATEnsemble] Loaded GAT Platt calibrator from: {calibrator_path}")
        else:
            default_cal = Path(model_path).parent / "gat_calibrator.json"
            if default_cal.exists():
                self.gat_calibrator = PlattScaler.load(str(default_cal))
                print(f"[GATEnsemble] Loaded GAT Platt calibrator from default: {default_cal}")
