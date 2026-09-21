"""
Stage 3: ONE-TIME test evaluation of the single validation-selected candidate.
Run exactly once. Uses the validation-selected threshold; nothing is tuned here.
Base predictions follow the locked protocol (full base checkpoints, full train graph), same as scripts/final_evaluation.py.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR, PROJECT_ROOT
from src.models.ensemble_model import PPIEnsemble
from src.utils.bio_encoder import BioFeatureEncoder
from src.analysis.biological_managers import BiologicalManager, ensemble_bio_score
from src.training.train_ensemble import predict_sequence_model, predict_graph_model, load_base_models

EXP_DIR = PROJECT_ROOT / "models" / "experiments"
LOG = PROJECT_ROOT / "assets" / "evaluation" / "experiment_log.json"
BASELINE = {"f1": 0.9155, "roc_auc": 0.9626, "pr_auc": 0.9696, "accuracy": 0.9175}


def main():
    meta = json.loads((EXP_DIR / "ensemble_candidate.json").read_text())
    assert meta["feature_set"] == "base_8", "this script only supports the selected base_8 candidate"
    log = json.loads(LOG.read_text())
    assert "final_test" not in log, "final test already performed; refusing to run a second time"

    print("FINAL CANDIDATE SELECTED — BEGINNING ONE-TIME TEST EVALUATION", flush=True)
    device = torch.device("cpu")
    bio_mapping = BioFeatureEncoder().get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values())))
    emb = {k: v.float() for k, v in torch.load(PROCESSED_DATA_DIR / "embeddings.pt", weights_only=False).items()}
    nm = torch.load(PROCESSED_DATA_DIR / "ppi_graph_mapping.pt", weights_only=False)
    g = torch.load(PROCESSED_DATA_DIR / "ppi_graph.pt", weights_only=False).to(device)
    df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    n_all = len(df)
    df = df[df.protein1.isin(nm) & df.protein2.isin(nm) & df.protein1.isin(emb) & df.protein2.isin(emb)].reset_index(drop=True)
    print(f"test rows evaluated: {len(df)}/{n_all}", flush=True)

    seq_m, gnn_m = load_base_models(MODELS_DIR / "sequence_model_best.pth", MODELS_DIR / "graph_model_best.pth",
                                    g, next(iter(emb.values())).shape[-1], g.x.shape[1], device)
    sp = predict_sequence_model(seq_m, emb, bio_mapping, df.protein1.values, df.protein2.values, device, bio_dim=bio_dim)
    gp = predict_graph_model(gnn_m, g, nm, df.protein1.values, df.protein2.values, device)
    bm = BiologicalManager()
    bio = np.array([ensemble_bio_score(bm, a, b) for a, b in zip(df.protein1, df.protein2)], dtype=np.float32).reshape(-1, 1)

    model = joblib.load(EXP_DIR / "ensemble_candidate.pkl")
    p = model.predict_proba(PPIEnsemble._build_features(sp, gp, bio))[:, 1]
    y = df.label.values
    yh = (p >= meta["threshold"]).astype(int)
    res = {"threshold_from_validation": meta["threshold"], "n_test": int(len(df)),
           "accuracy": float(accuracy_score(y, yh)), "precision": float(precision_score(y, yh)),
           "recall": float(recall_score(y, yh)), "f1": float(f1_score(y, yh)),
           "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
    print(json.dumps(res, indent=2))
    print("\nMetric      Candidate   Baseline    Diff")
    diffs = {}
    for k in ["accuracy", "f1", "roc_auc", "pr_auc"]:
        diffs[k] = res[k] - BASELINE[k]
        print(f"{k:10s}  {res[k]:.4f}     {BASELINE[k]:.4f}     {diffs[k]:+.4f}")

    log["final_test"] = {"experiment_id": meta["experiment_id"], "performed_once": True,
                         "metrics": res, "locked_baseline": BASELINE, "absolute_difference": diffs,
                         "note": "One-time evaluation after validation-based selection; no tuning followed."}
    LOG.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
