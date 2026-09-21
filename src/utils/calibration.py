import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


def ece_score(y_true, p, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(total)


def logit(p, eps: float = 1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattScaler:
    """Platt scaling: p_cal = sigmoid(a * logit + b), fit by logistic regression on VALIDATION logits only."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = float(a), float(b)

    def fit(self, logits, y):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), np.asarray(y).astype(int))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform_logits(self, logits):
        z = self.a * np.asarray(logits, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def transform_probs(self, p):
        """Calibrate raw sigmoid probabilities (converted back to logits first)."""
        return self.transform_logits(logit(p))

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"method": "platt", "a": self.a, "b": self.b}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["a"], d["b"])


def calibration_report(y, p_raw, p_cal) -> dict:
    y = np.asarray(y)
    return {
        "before": {"ECE": round(ece_score(y, p_raw), 5), "Brier": round(float(brier_score_loss(y, p_raw)), 5),
                   "accuracy@0.5": round(float(((np.asarray(p_raw) > 0.5) == y).mean()), 5)},
        "after": {"ECE": round(ece_score(y, p_cal), 5), "Brier": round(float(brier_score_loss(y, p_cal)), 5),
                  "accuracy@0.5": round(float(((np.asarray(p_cal) > 0.5) == y).mean()), 5)},
    }


def cross_fitted_probs(logits, y, n_splits: int = 5, seed: int = 42):
    """Out-of-sample calibrated probabilities on the validation set (each fold calibrated by a scaler fit on the rest)."""
    logits, y = np.asarray(logits), np.asarray(y)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(logits, y):
        out[te] = PlattScaler().fit(logits[tr], y[tr]).transform_logits(logits[te])
    return out
