"""
Builds the PREDICTED protein interaction network used by the therapeutic-target analysis.

Network = known interactions (train.csv positives, the edges the models were trained on)
        + interactions PREDICTED by the production ensemble (ESM-2 + GraphSAGE + XGBoost) on the
          held-out pairs of val.csv and test.csv, at the validation-selected threshold from
          assets/evaluation/final_test_metrics.json.

Held-out pairs are not part of the training graph (graph_construction.py uses training positives only),
so every predicted edge is an interaction the model inferred rather than one it was given. Each predicted
edge keeps its ground-truth label: label 1 = confirmed by STRING, label 0 = not a STRING interaction
(a candidate novel interaction or a false positive; the model cannot tell these apart).

Scoring reuses get_model_predictions() from src/evaluation/compare_models.py, the exact routine behind the
reported test metrics, and is checked against the committed test predictions in
ESMGAT/results/final_predictions.csv.

Outputs:
  data/processed/predicted_network.csv               protein1, protein2, source, probability, split, label
  assets/evaluation/predicted_network_summary.json   edge counts and the reproduction check

Run from the repo root:  python scripts/build_predicted_network.py
"""
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, MODELS_DIR
from src.utils.esm_config import ESM_EMBED_DIM
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor, GINLinkPredictor
from src.evaluation.compare_models import get_model_predictions

ENSEMBLE_KEY = "Full Ensemble (XGBoost)"


def load_models(device):
    embeddings = torch.load(PROCESSED_DATA_DIR / "embeddings.pt", map_location="cpu", weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}  # as compare_models.py
    node_mapping = torch.load(PROCESSED_DATA_DIR / "ppi_graph_mapping.pt", map_location="cpu", weights_only=False)
    graph_data = torch.load(PROCESSED_DATA_DIR / "ppi_graph.pt", map_location="cpu", weights_only=False)

    seq_model = SequencePPIModel(input_dim=ESM_EMBED_DIM, hidden_dim=1024).to(device)
    seq_model.load_state_dict(torch.load(MODELS_DIR / "sequence_model_best.pth", map_location=device))
    seq_model.eval()

    state_dict = torch.load(MODELS_DIR / "graph_model_best.pth", map_location=device, weights_only=False)
    in_channels = graph_data.x.shape[1]
    if any("convs" in k for k in state_dict.keys()):
        graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
    else:
        graph_model = SAGELinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
    graph_model.load_state_dict(state_dict)
    graph_model.eval()

    ensemble_model = joblib.load(MODELS_DIR / "ensemble_model.pkl")
    return embeddings, node_mapping, graph_data, seq_model, graph_model, ensemble_model


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metrics = json.loads((PROJECT_ROOT / "assets" / "evaluation" / "final_test_metrics.json").read_text())
    threshold = float(metrics["models"][ENSEMBLE_KEY]["val_selected_threshold"])
    print(f"Ensemble threshold (selected on val.csv): {threshold:.2f}")

    embeddings, node_mapping, graph_data, seq_model, graph_model, ensemble_model = load_models(device)

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    known = train_df[train_df["label"] == 1][["protein1", "protein2"]].copy()
    known["source"] = "known"
    known["probability"] = np.nan
    known["split"] = "train"
    known["label"] = 1

    predicted_parts, reproduction = [], {}
    for split in ("val", "test"):
        df = pd.read_csv(PROCESSED_DATA_DIR / f"{split}.csv")
        labels, _, _, ens_preds, _, _, scored_df = get_model_predictions(
            df, seq_model, graph_model, ensemble_model, None, embeddings, None, None,
            node_mapping, graph_data, device, desc=f"{split} scoring",
        )
        scored_df = scored_df.assign(probability=ens_preds, split=split)
        print(f"{split}: scored {len(scored_df)}/{len(df)} pairs")

        if split == "test":
            ref = pd.read_csv(PROJECT_ROOT / "ESMGAT" / "results" / "final_predictions.csv")
            merged = scored_df.merge(ref[["protein1", "protein2", "p_graphsage_ensemble"]], on=["protein1", "protein2"])
            max_diff = float(np.abs(merged["probability"] - merged["p_graphsage_ensemble"]).max())
            reproduction = {"reference": "ESMGAT/results/final_predictions.csv", "pairs_compared": int(len(merged)),
                            "max_abs_probability_diff": max_diff}
            print(f"Reproduction check vs committed test predictions: {len(merged)} pairs, max |diff| = {max_diff:.2e}")
            if max_diff > 1e-4:
                raise SystemExit("Scoring does not reproduce the committed test predictions; refusing to write the network.")

        predicted_parts.append(scored_df[scored_df["probability"] >= threshold][
            ["protein1", "protein2", "probability", "split", "label"]].assign(source="predicted"))

    predicted = pd.concat(predicted_parts, ignore_index=True)
    network = pd.concat([known, predicted[known.columns]], ignore_index=True)
    out_csv = PROCESSED_DATA_DIR / "predicted_network.csv"
    network.to_csv(out_csv, index=False)

    nodes_known = set(known["protein1"]) | set(known["protein2"])
    nodes_all = set(network["protein1"]) | set(network["protein2"])
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "known train.csv positives + held-out (val/test) pairs predicted positive by the production "
                      "ESM-2 + GraphSAGE + XGBoost ensemble at the val-selected threshold",
        "threshold": threshold,
        "known_edges": int(len(known)),
        "predicted_edges": int(len(predicted)),
        "predicted_edges_confirmed_by_string": int((predicted["label"] == 1).sum()),
        "predicted_edges_not_in_string": int((predicted["label"] == 0).sum()),
        "predicted_edges_by_split": predicted["split"].value_counts().to_dict(),
        "total_edges": int(len(network)),
        "nodes_known_network": len(nodes_known),
        "nodes_predicted_network": len(nodes_all),
        "reproduction_check": reproduction,
    }
    out_json = PROJECT_ROOT / "assets" / "evaluation" / "predicted_network_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
