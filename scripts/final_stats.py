import torch
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.evaluation.metrics_reporter import report_all_metrics
from src.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.bio_encoder import BioFeatureEncoder

def main():
    device = torch.device("cpu") # Keep on CPU for stability in this script
    
    val_path = PROCESSED_DATA_DIR / "val.csv"
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    bio_path = PROCESSED_DATA_DIR / "bio_metadata_cache.csv"

    val_df = pd.read_csv(val_path)
    embeddings = torch.load(emb_path, weights_only=False)
    embeddings = {k: v.float().cpu() if v.dtype == torch.float16 else v.cpu() for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)
    graph_data = torch.load(graph_path, weights_only=False).to(device)
    bio_df = pd.read_csv(bio_path).set_index("protein_id")["localization"].to_dict() if bio_path.exists() else {}

    filtered_df = val_df[
        val_df["protein1"].isin(embeddings) & 
        val_df["protein2"].isin(embeddings) &
        val_df["protein1"].isin(node_mapping) &
        val_df["protein2"].isin(node_mapping)
    ].copy()
    
    y_true = filtered_df["label"].values
    
    # Load bio-features for dimension detection
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0
    
    # Dynamically detect input_dim
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1]
    print(f"Detected Dimensions: Sequence={input_dim}, Biology={bio_dim}")

    seq_model = SequencePPIModel(input_dim=input_dim, bio_dim=bio_dim).to(device)
    seq_model.load_state_dict(torch.load(MODELS_DIR / "sequence_model_best.pth", map_location=device))
    seq_model.eval()

    from torch_geometric.utils import degree
    deg = degree(graph_data.edge_index[0], graph_data.x.shape[0]).view(-1, 1)
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-6)
    graph_data.x = torch.cat([graph_data.x, deg_norm], dim=-1)

    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128).to(device)
    graph_model.load_state_dict(torch.load(MODELS_DIR / "graph_model_best.pth", map_location=device))
    graph_model.eval()

    ensemble = PPIEnsemble(str(MODELS_DIR / "ensemble_model.pkl"))

    batch_emb1 = []
    batch_emb2 = []
    g_src = []
    g_dst = []
    bio_features = []
    
    for _, row in filtered_df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        e1, e2 = embeddings[p1], embeddings[p2]
        batch_emb1.append(e1.mean(dim=0) if e1.dim() > 1 else e1)
        batch_emb2.append(e2.mean(dim=0) if e2.dim() > 1 else e2)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        loc1, loc2 = bio_df.get(p1, "unk1"), bio_df.get(p2, "unk2")
        bio_features.append([1.0 if loc1 == loc2 and loc1 != "unk1" else 0.0])

    batch_emb1 = torch.stack(batch_emb1)
    batch_emb2 = torch.stack(batch_emb2)
    bio_features = np.array(bio_features)
    
    seq_probs = []
    with torch.no_grad():
        for i in range(0, len(batch_emb1), 64):
            e1 = batch_emb1[i:i+64].to(device)
            e2 = batch_emb2[i:i+64].to(device)
            out = seq_model(e1, e2)
            seq_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    seq_probs = np.array(seq_probs)

    g_edge_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    graph_probs = []
    with torch.no_grad():
        for i in range(0, g_edge_index.size(1), 10000):
            chunk = g_edge_index[:, i:i+10000]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            graph_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    graph_probs = np.array(graph_probs)

    ensemble_probs = ensemble.predict(seq_probs, graph_probs, bio_features=bio_features, method="stacking")

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    print(f"RESULTS_ACC: {accuracy_score(y_true, ensemble_probs.round())}")
    print(f"RESULTS_F1: {f1_score(y_true, ensemble_probs.round())}")
    print(f"RESULTS_AUC: {roc_auc_score(y_true, ensemble_probs)}")
    print(f"GRAPH_F1: {f1_score(y_true, graph_probs.round())}")

if __name__ == "__main__":
    main()
