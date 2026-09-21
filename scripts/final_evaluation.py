import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.evaluation.metrics_reporter import MetricsReporter
from src.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR, PROJECT_ROOT
from src.utils.bio_encoder import BioFeatureEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Final Evaluation on {device}...")

    # 1. Load Data & Supports
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"

    if not all(p.exists() for p in [val_path, test_path, emb_path, map_path, graph_path]):
        print("Missing required data files (val.csv, test.csv, embeddings.pt, mapping, or graph).")
        return

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    embeddings = torch.load(emb_path, weights_only=False)
    # Convert half to float
    embeddings = {k: v.float().cpu() if v.dtype == torch.float16 else v.cpu() for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)
    graph_data = torch.load(graph_path, weights_only=False).to(device)

    def filter_split(df):
        return df[
            df["protein1"].isin(node_mapping) &
            df["protein2"].isin(node_mapping)
        ].copy()

    # Protocol: val.csv -> threshold selection ONLY; test.csv -> final reported metrics ONLY
    filtered_val = filter_split(val_df)
    filtered_test = filter_split(test_df)
    
    # 2. Load Models
    print("Loading Models...")
    
    # Dynamically detect input_dim
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1]
    
    # Load bio-features for dimension detection
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0
    print(f"Detected Dimensions: Sequence={input_dim}, Biology={bio_dim}")

    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    seq_model.load_state_dict(torch.load(MODELS_DIR / "sequence_model_best.pth", map_location=device))
    seq_model.eval()

    # Graph features are already injected properly by extract_topo in run_pipeline.py

    from src.models.graph_model import GINLinkPredictor
    state_dict = torch.load(MODELS_DIR / "graph_model_best.pth", map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    
    if is_gin:
        graph_model = GINLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128).to(device)
    else:
        graph_model = SAGELinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=256).to(device)
        
    graph_model.load_state_dict(state_dict)
    graph_model.eval()

    ensemble = PPIEnsemble(str(MODELS_DIR / "ensemble_model.pkl"))

    def predict_split(split_df):
    
        batch_emb1 = []
        batch_emb2 = []
        g_src = []
        g_dst = []
    
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Preparing Data"):
            p1, p2 = row["protein1"], row["protein2"]
            e1, e2 = embeddings[p1], embeddings[p2]
            e1_mean = e1.mean(dim=0) if e1.dim() > 1 else e1
            e2_mean = e2.mean(dim=0) if e2.dim() > 1 else e2
        
            if bio_mapping:
                b1 = bio_mapping.get(p1, torch.zeros(bio_dim))
                b2 = bio_mapping.get(p2, torch.zeros(bio_dim))
                e1_mean = torch.cat([e1_mean, b1])
                e2_mean = torch.cat([e2_mean, b2])
            
            batch_emb1.append(e1_mean)
            batch_emb2.append(e2_mean)
            g_src.append(node_mapping[p1])
            g_dst.append(node_mapping[p2])


        batch_emb1 = torch.stack(batch_emb1)
        batch_emb2 = torch.stack(batch_emb2)
    
        seq_probs = []
        with torch.no_grad():
            for i in range(0, len(batch_emb1), 64):
                e1 = batch_emb1[i:i+64].to(device)
                e2 = batch_emb2[i:i+64].to(device)
                out = seq_model(e1, e2)
                seq_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
        seq_probs = np.array(seq_probs)

        # Graph Model
        g_edge_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
        graph_probs = []
        with torch.no_grad():
            for i in range(0, g_edge_index.size(1), 10000):
                chunk = g_edge_index[:, i:i+10000]
                out = graph_model(graph_data.x, graph_data.edge_index, chunk)
                graph_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
        graph_probs = np.array(graph_probs)

        # Ensemble
        ensemble_probs = ensemble.predict(seq_probs, graph_probs, method="stacking")

        return seq_probs, graph_probs, ensemble_probs

    # 3/4. Generate predictions, select thresholds on val, report on test
    print("Generating predictions (validation: threshold selection only)...")
    val_seq, val_graph, val_ens = predict_split(filtered_val)
    y_val = filtered_val["label"].values
    print("Generating predictions (test: final evaluation)...")
    test_seq, test_graph, test_ens = predict_split(filtered_test)
    y_test = filtered_test["label"].values

    print(f"\nValidation rows: {len(filtered_val)}/{len(val_df)} | Test rows evaluated: {len(filtered_test)}/{len(test_df)}")

    reporter = MetricsReporter()
    names = ["ESM-MLP", "GraphSAGE", "Ensemble"]
    val_probs = [val_seq, val_graph, val_ens]
    test_probs = [test_seq, test_graph, test_ens]
    rows = []
    for name, vp, tp in zip(names, val_probs, test_probs):
        thresh, _ = reporter.find_best_f1_threshold(y_val, vp)  # val.csv only
        m = reporter.calculate_metrics(y_test, tp, thresh)       # test.csv only
        m["Model"] = name
        m["Val Thresh"] = thresh
        rows.append(m)
    reporter.print_table(
        "FINAL TEST EVALUATION (test.csv; thresholds selected on val.csv)",
        rows, ["Model", "Val Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    )

if __name__ == "__main__":
    main()
