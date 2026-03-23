import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import numpy as np
import pandas as pd
from src.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GINLinkPredictor, GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.data.feature_extraction import ESMFeatureExtractor
from src.utils.bio_encoder import BioFeatureEncoder
from src.analysis.explainability import PPIExplainer
from src.analysis.biological_managers import BiologicalManager

def analyze_case(p1, p2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Get Base Probabilities
    # (Simplified version of test_fp.py logic)
    esm = ESMFeatureExtractor(device=device)
    bio_mgr = BiologicalManager()
    bio_enc = BioFeatureEncoder()
    
    seq_mgr = torch.load(PROCESSED_DATA_DIR / "embeddings.pt", weights_only=False)
    e1 = seq_mgr[p1].unsqueeze(0).to(device).float()
    e2 = seq_mgr[p2].unsqueeze(0).to(device).float()
    
    bio_meta = bio_mgr.get_bio_metadata([p1, p2])
    def get_loc(pid):
        row = bio_meta[bio_meta["protein_id"] == pid]
        return row.iloc[0]["localization"] if not row.empty else ""
    
    loc1, loc2 = get_loc(p1), get_loc(p2)
    b1 = bio_enc.encode_protein(loc1).unsqueeze(0).to(device)
    b2 = bio_enc.encode_protein(loc2).unsqueeze(0).to(device)
    
    seq_model = SequencePPIModel(input_dim=480).to(device)
    seq_model.load_state_dict(torch.load(MODELS_DIR / "sequence_model_best.pth", map_location=device))
    seq_model.eval()
    
    with torch.no_grad():
        s_prob = torch.sigmoid(seq_model(torch.cat([e1, b1], dim=1), torch.cat([e2, b2], dim=1))).item()
        
    # Graph Prob
    graph_data = torch.load(PROCESSED_DATA_DIR / "ppi_graph.pt", weights_only=False).to(device)
    mapping = torch.load(PROCESSED_DATA_DIR / "ppi_graph_mapping.pt", weights_only=False)
    
    state_dict = torch.load(MODELS_DIR / "graph_model_best.pth", map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    
    in_channels = graph_data.x.shape[1]
    if is_gin:
        graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
    else:
        graph_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=128, heads=4).to(device)
        
    graph_model.load_state_dict(state_dict)
    graph_model.eval()
    
    g_prob = 0.5
    if p1 in mapping and p2 in mapping:
        idx1, idx2 = mapping[p1], mapping[p2]
        with torch.no_grad():
            g_out = graph_model(graph_data.x, graph_data.edge_index, torch.tensor([[idx1], [idx2]]).to(device))
            g_prob = torch.sigmoid(g_out).item()
            
    # Ensemble
    ens = PPIEnsemble(str(MODELS_DIR / "ensemble_model.pkl"))
    bio_score = 1.0 if loc1 == loc2 and loc1 != "" else 0.0
    
    # 2. SHAP Explanation
    explainer = PPIExplainer(str(MODELS_DIR / "ensemble_model.pkl"))
    
    # Features for the ensemble
    conf_s = abs(s_prob - 0.5)
    conf_g = abs(g_prob - 0.5)
    disagreement = abs(s_prob - g_prob)
    max_conf = max(conf_s, conf_g)
    
    shap_vals = explainer.explain_prediction(
        s_prob, g_prob, conf_s, conf_g, disagreement, max_conf, bio_score
    )
    
    print("\n--- Deep Analysis ---")
    print(f"Pair: {p1} - {p2}")
    print(f"Sequence Prob: {s_prob:.4f}")
    print(f"Graph Prob: {g_prob:.4f}")
    print(f"Bio Score: {bio_score}")
    print(f"Ensemble Prob: {ens.predict(np.array([s_prob]), np.array([g_prob]), np.array([[bio_score]]))[0]:.4f}")
    
    print("\nSHAP Contributions (Positive means pushing towards Interaction):")
    features = ["ESM-MLP", "GAT", "Conf-Seq", "Conf-GAT", "Disagreement", "MaxConf", "Bio-Loc"]
    for name, val in zip(features, shap_vals[0]):
        print(f"{name:12}: {val:+.4f}")

if __name__ == "__main__":
    analyze_case("ENSP00000263694", "ENSP00000436786")
