"""
Audit measurements for the PPI dataset and (optionally) the trained meta-learner.

Reports, as JSON:
  * negative-label contamination: fraction of negatives that are real STRING interactions
    (score > 0 = any confidence, and score >= 400 / >= 700 for context)
  * degree-bias baseline: accuracy/AUC of a classifier that sees ONLY node degree (positive-set degree)
  * meta-feature mean |SHAP| for the XGBoost meta-learner   (needs cached val features + matching model)
  * calibration (ECE, Brier) of the seq / graph / ensemble validation probabilities (same requirement)

Model-based metrics are only meaningful when models were trained on the CURRENT splits, so they are
skipped with --skip-model-metrics (used right after re-running preprocessing, before retraining).

Usage:
  python scripts/audit_measurements.py --out assets/evaluation/audit/audit_before.json
  python scripts/audit_measurements.py --out assets/evaluation/audit/audit_after.json --skip-model-metrics
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.paths import PROCESSED_DATA_DIR, STRING_FILE, MODELS_DIR, PROJECT_ROOT


def canon(p1, p2):
    return np.where(p1 < p2, p1, p2), np.where(p1 < p2, p2, p1)


def load_string_scores():
    df = pd.read_csv(STRING_FILE, sep=" ", usecols=["protein1", "protein2", "combined_score"])
    df["protein1"] = df["protein1"].astype(str).str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].astype(str).str.replace("9606.", "", regex=False)
    a, b = canon(df["protein1"].values, df["protein2"].values)
    out = pd.DataFrame({"protein1": a, "protein2": b, "score": df["combined_score"].values})
    return out.drop_duplicates(subset=["protein1", "protein2"])


def contamination(splits, string_df):
    key = string_df["protein1"] + "|" + string_df["protein2"]
    score_of = dict(zip(key, string_df["score"]))
    res = {}
    all_neg = 0
    hit = {"any": 0, "ge400": 0, "ge700": 0}
    for name, df in splits.items():
        neg = df[df["label"] == 0]
        a, b = canon(neg["protein1"].values, neg["protein2"].values)
        sc = np.array([score_of.get(x + "|" + y, 0) for x, y in zip(a, b)])
        r = {"n_negatives": int(len(neg)),
             "contaminated_any_score": int((sc > 0).sum()),
             "contaminated_ge400": int((sc >= 400).sum()),
             "contaminated_ge700": int((sc >= 700).sum())}
        r["contamination_pct_any"] = round(100 * r["contaminated_any_score"] / max(1, len(neg)), 3)
        res[name] = r
        all_neg += len(neg)
        hit["any"] += r["contaminated_any_score"]
        hit["ge400"] += r["contaminated_ge400"]
        hit["ge700"] += r["contaminated_ge700"]
    res["ALL"] = {"n_negatives": all_neg, "contaminated_any_score": hit["any"],
                  "contaminated_ge400": hit["ge400"], "contaminated_ge700": hit["ge700"],
                  "contamination_pct_any": round(100 * hit["any"] / max(1, all_neg), 3)}
    return res


def degree_baseline(splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    all_df = pd.concat(splits.values(), ignore_index=True)
    pos = all_df[all_df["label"] == 1]
    deg = pd.concat([pos["protein1"], pos["protein2"]]).value_counts().to_dict()

    def feats(df):
        d1 = np.log1p(df["protein1"].map(deg).fillna(0).values)
        d2 = np.log1p(df["protein2"].map(deg).fillna(0).values)
        return np.column_stack([np.minimum(d1, d2), np.maximum(d1, d2), d1 + d2, d1 * d2])

    Xtr, ytr = feats(splits["train"]), splits["train"]["label"].values
    Xte, yte = feats(splits["test"]), splits["test"]["label"].values
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {"description": "logistic regression on log positive-degree of the two proteins only; fit on train, scored on test",
            "test_accuracy": round(float(accuracy_score(yte, p > 0.5)), 4),
            "test_roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "majority_class_accuracy": round(float(max(yte.mean(), 1 - yte.mean())), 4)}


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def model_metrics():
    import joblib
    import shap
    from sklearn.metrics import brier_score_loss
    from src.models.ensemble_model import PPIEnsemble

    cache = np.load(MODELS_DIR / "experiments" / "features_cache.npz")
    y = cache["va_y"]
    out = {"calibration_val": {}}
    for name, key in [("sequence", "va_seq"), ("graph", "va_graph")]:
        p = cache[key].astype(float)
        out["calibration_val"][name] = {"ECE": round(ece(y, p), 4), "Brier": round(float(brier_score_loss(y, p)), 4)}

    meta = joblib.load(MODELS_DIR / "ensemble_model.pkl")
    n_feat = meta.n_features_in_
    bio = cache["va_bio"].reshape(-1, 1) if n_feat == 8 else None
    X = PPIEnsemble._build_features(cache["va_seq"], cache["va_graph"], bio)
    names = ["p_seq", "p_graph", "conf_seq", "conf_graph", "diff", "max_conf", "consensus", "bio_score"][:X.shape[1]]
    pe = meta.predict_proba(X)[:, 1]
    out["calibration_val"]["ensemble"] = {"ECE": round(ece(y, pe), 4), "Brier": round(float(brier_score_loss(y, pe)), 4)}
    sv = shap.TreeExplainer(meta).shap_values(X)
    out["meta_feature_mean_abs_shap"] = {n: round(float(v), 5) for n, v in zip(names, np.abs(sv).mean(0))}
    out["meta_feature_xgb_gain"] = {n: round(float(v), 5) for n, v in zip(names, meta.feature_importances_)}
    if n_feat == 8:
        out["bio_score_pct_default_0.5_val"] = round(float((np.abs(cache["va_bio"] - 0.5) < 1e-9).mean() * 100), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-model-metrics", action="store_true")
    args = ap.parse_args()

    splits = {n: pd.read_csv(PROCESSED_DATA_DIR / f"{n}.csv") for n in ["train", "val", "test"]}
    string_df = load_string_scores()
    res = {"split_sizes": {n: int(len(d)) for n, d in splits.items()},
           "negative_contamination": contamination(splits, string_df),
           "degree_bias_baseline": degree_baseline(splits)}
    if args.skip_model_metrics:
        res["model_metrics"] = "skipped: models must be retrained on the new splits (Colab) before SHAP/ECE are meaningful"
    else:
        res["model_metrics"] = model_metrics()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
