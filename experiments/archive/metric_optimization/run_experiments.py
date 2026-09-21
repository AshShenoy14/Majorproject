"""
Stage 2: XGBoost meta-learner experiments (Exp 1 hyperparameters, Exp 2 topology, Exp 3 combined).

Uses ONLY models/experiments/features_cache.npz (train OOF + val features). test.csv is never touched.
Selection: highest VALIDATION F1 (threshold chosen on validation). Nothing here writes test metrics.
Never writes models/ensemble_model.pkl; the selected candidate goes to models/experiments/ensemble_candidate.pkl.
"""
import os
import sys
import json
import time
import datetime as dt

import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.paths import PROJECT_ROOT, MODELS_DIR
from src.models.ensemble_model import PPIEnsemble
from src.evaluation.metrics_reporter import MetricsReporter

EXP_DIR = PROJECT_ROOT / "models" / "experiments"
CACHE = EXP_DIR / "features_cache.npz"
LEAK = EXP_DIR / "leakage_report.json"
LOG = PROJECT_ROOT / "assets" / "evaluation" / "experiment_log.json"
CANDIDATE = EXP_DIR / "ensemble_candidate.pkl"
CANDIDATE_META = EXP_DIR / "ensemble_candidate.json"
N_RANDOM = 60
SEED = 42

GRID = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 400, 600, 800],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.01, 0.1, 1.0],
    "reg_lambda": [0.01, 0.1, 1.0],
}
# Baseline meta-learner settings (src/models/ensemble_model.py); reg_lambda is xgboost's default (1.0)
DEFAULT_HP = dict(max_depth=7, learning_rate=0.01, n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0)
FIXED = dict(gamma=1, objective="binary:logistic", eval_metric="logloss", random_state=SEED, n_jobs=-1)


def feats(d, prefix, topo):
    X = PPIEnsemble._build_features(d[f"{prefix}_seq"], d[f"{prefix}_graph"], d[f"{prefix}_bio"].reshape(-1, 1))
    if topo:
        X = np.column_stack((X, d[f"{prefix}_cn"], d[f"{prefix}_jac"], d[f"{prefix}_prd"]))
    return X


def val_metrics(y, p):
    thr, m = MetricsReporter.find_best_f1_threshold(y, p)  # validation-only threshold sweep 0.10-0.90
    return {"threshold": float(thr), "accuracy": float(m["Accuracy"]), "precision": float(m["Precision"]),
            "recall": float(m["Recall"]), "f1": float(m["F1"]), "roc_auc": float(m["ROC-AUC"]),
            "pr_auc": float(m["PR-AUC"])}


def main():
    d = np.load(CACHE)
    leak = json.loads(LEAK.read_text())
    ytr, yva = d["tr_y"], d["va_y"]
    X = {False: (feats(d, "tr", False), feats(d, "va", False)), True: (feats(d, "tr", True), feats(d, "va", True))}
    BASE_NAMES = ["seq_prob", "graph_prob", "conf_seq", "conf_graph", "disagreement", "max_conf", "consensus", "bio_score"]
    TOPO_NAMES = BASE_NAMES + ["common_neighbors", "jaccard", "pagerank_delta"]

    cv_cfg = ("Base-model OOF: 5-fold StratifiedKFold(shuffle, rs=42) on train.csv, fold-specific graphs. "
              "Meta-learner: fit once on the full train OOF feature matrix; evaluated on val.csv.")
    records, models = [], {}
    n = [0]

    def run(exp, name, hp, topo, note):
        n[0] += 1
        eid = f"{exp}-{n[0]:03d}"
        t0 = time.time()
        m = xgb.XGBClassifier(**hp, **FIXED)
        m.fit(X[topo][0], ytr)
        p = m.predict_proba(X[topo][1])[:, 1]
        met = val_metrics(yva, p)
        rec = {"experiment_id": eid, "name": name, "experiment_group": exp,
               "datetime": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "model_configuration": {**hp, **{k: v for k, v in FIXED.items() if k != "n_jobs"}},
               "feature_set": "topology_11" if topo else "base_8",
               "feature_names": TOPO_NAMES if topo else BASE_NAMES,
               "cv_configuration": cv_cfg,
               "validation_metrics": met, "threshold": met["threshold"], "runtime_seconds": round(time.time() - t0, 2),
               "note": note, "leakage_checks": None, "selected": False, "reason": None,
               "model_artifact_path": None}
        records.append(rec)
        models[eid] = m
        print(f"{eid} {name:28s} F1={met['f1']:.4f} AUC={met['roc_auc']:.4f} PR={met['pr_auc']:.4f} thr={met['threshold']:.3f} ({rec['runtime_seconds']}s)", flush=True)
        return rec

    # ---- reference: locked baseline model scored on the same validation features (val only) ----
    import joblib as jl
    base_model = jl.load(MODELS_DIR / "ensemble_model.pkl")
    pb = base_model.predict_proba(X[False][1])[:, 1]
    bm = val_metrics(yva, pb)
    ref = {"experiment_id": "REF-000", "name": "locked_baseline_ensemble_on_val", "experiment_group": "reference",
           "datetime": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "model_configuration": "models/ensemble_model.pkl (unchanged, read-only)", "feature_set": "base_8",
           "feature_names": BASE_NAMES, "cv_configuration": "n/a (pre-existing model)",
           "validation_metrics": bm, "threshold": bm["threshold"], "runtime_seconds": 0,
           "leakage_checks": None, "selected": False, "reason": None, "model_artifact_path": "models/ensemble_model.pkl"}
    records.append(ref)
    print(f"REF locked baseline on val: F1={bm['f1']:.4f} AUC={bm['roc_auc']:.4f} PR={bm['pr_auc']:.4f} thr={bm['threshold']:.3f}", flush=True)

    # ---- Exp 0/2: control vs topology at the baseline's hyperparameters, same OOF features ----
    c0 = run("EXP2", "base8_default_hp (control)", DEFAULT_HP, False, "control: baseline hyperparameters refit on the regenerated leak-free OOF features")
    c2 = run("EXP2", "topo11_default_hp", DEFAULT_HP, True, "base 8 + CN + Jaccard + PageRank-delta at baseline hyperparameters")

    # ---- Exp 1: randomized search on base features ----
    grid_size = int(np.prod([len(v) for v in GRID.values()]))
    combos = list(ParameterSampler(GRID, n_iter=N_RANDOM, random_state=SEED))
    print(f"Exp1: {N_RANDOM} random configs of {grid_size} in grid", flush=True)
    e1 = [run("EXP1", "base8_random_search", hp, False, f"random search {N_RANDOM}/{grid_size}") for hp in combos]
    best1 = max(e1, key=lambda r: r["validation_metrics"]["f1"])

    # ---- Exp 3: topology with best (and top-5) Exp1 hyperparameters ----
    topo_useful = c2["validation_metrics"]["f1"] > c0["validation_metrics"]["f1"]
    top5 = sorted(e1, key=lambda r: -r["validation_metrics"]["f1"])[:5]
    e3 = [run("EXP3", "topo11_top5_exp1_hp", {k: r["model_configuration"][k] for k in GRID}, True,
              f"topology features + hyperparameters of {r['experiment_id']}") for r in top5]
    # Exp3 is run regardless so the topology verdict is not decided by a single default-hp comparison

    # ---- selection: highest validation F1; must strictly beat the locked baseline's validation F1 ----
    cands = [r for r in records if r["experiment_group"] != "reference"]
    best = max(cands, key=lambda r: r["validation_metrics"]["f1"])
    improved = best["validation_metrics"]["f1"] > bm["f1"]
    ties = [r for r in cands if r["validation_metrics"]["f1"] == best["validation_metrics"]["f1"]]

    for r in records:
        r["leakage_checks"] = leak
        if r["experiment_group"] == "reference":
            r["reason"] = "Reference row (locked baseline scored on validation only); not a candidate."
        elif r is best and improved:
            r["selected"] = True
            r["reason"] = (f"Highest validation F1 ({r['validation_metrics']['f1']:.4f}) among {len(cands)} candidates, "
                           f"> locked baseline val F1 {bm['f1']:.4f}.")
        else:
            gap = best["validation_metrics"]["f1"] - r["validation_metrics"]["f1"]
            r["reason"] = f"Not selected: validation F1 {gap:.4f} below best candidate {best['experiment_id']}."

    if improved:
        b = models[best["experiment_id"]]
        joblib.dump(b, CANDIDATE)
        best["model_artifact_path"] = "models/experiments/ensemble_candidate.pkl"
        CANDIDATE_META.write_text(json.dumps({
            "experiment_id": best["experiment_id"], "feature_set": best["feature_set"],
            "feature_names": best["feature_names"], "threshold": best["threshold"],
            "model_configuration": best["model_configuration"], "n_tied_at_best_f1": len(ties)}, indent=2))
    else:
        print("NO candidate improved validation F1 over the locked baseline; keeping existing ensemble.", flush=True)

    summary = {
        "n_experiments": len(cands), "grid_size": grid_size,
        "topology_vs_control_default_hp": {"base8": c0["validation_metrics"], "topo11": c2["validation_metrics"]},
        "best_base8_candidate": max([r for r in cands if r["feature_set"] == "base_8"], key=lambda r: r["validation_metrics"]["f1"])["experiment_id"],
        "best_topo11_candidate": max([r for r in cands if r["feature_set"] == "topology_11"], key=lambda r: r["validation_metrics"]["f1"])["experiment_id"],
        "selected_experiment_id": best["experiment_id"] if improved else None,
        "locked_baseline_val": bm,
        "protocol": "selection on validation F1 only; test.csv not accessed; no test metrics in this file at this stage",
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({"summary": summary, "experiments": records}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
