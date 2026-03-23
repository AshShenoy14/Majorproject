import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import numpy as np
import pandas as pd
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.sequence_manager import SequenceManager
from src.analysis.biological_managers import BiologicalManager
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GINLinkPredictor, GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.data.feature_extraction import ESMFeatureExtractor
from src.utils.bio_encoder import BioFeatureEncoder

def main():
    p1 = "ENSP00000265350"
    p2 = "ENSP00000326767"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seq_mgr = SequenceManager()
    bio_mgr = BiologicalManager()
    bio_enc = BioFeatureEncoder()
    esm = ESMFeatureExtractor(device=device)

    seqs = seq_mgr.get_sequences([p1, p2])
    print("Found Sequences:", len(seqs))

    embs = esm.get_embeddings(seqs, batch_size=2)
    e1 = embs[p1].unsqueeze(0).to(device).float()
    e2 = embs[p2].unsqueeze(0).to(device).float()

    seq_model = SequencePPIModel(input_dim=480).to(device)
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()

    with torch.no_grad():
        bio_meta = bio_mgr.get_bio_metadata([p1, p2])
        def get_encoded(pid):
            row = bio_meta[bio_meta["protein_id"] == pid]
            loc = row.iloc[0]["localization"] if not row.empty else ""
            return bio_enc.encode_protein(loc).to(device).float()

        b1 = get_encoded(p1).unsqueeze(0)
        b2 = get_encoded(p2).unsqueeze(0)
        e1_f = torch.cat([e1, b1], dim=1)
        e2_f = torch.cat([e2, b2], dim=1)
        
        seq_prob = torch.sigmoid(seq_model(e1_f, e2_f)).item()
        print("Sequence Model Prob:", seq_prob)

    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data = torch.load(graph_data_path, weights_only=False).to(device)
    mapping = torch.load(map_path, weights_only=False)

    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    state_dict = torch.load(graph_path, map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    
    in_channels = graph_data.x.shape[1]
    if is_gin:
        graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
    else:
        graph_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=128, heads=4).to(device)
    
    graph_model.load_state_dict(state_dict)
    graph_model.eval()

    graph_prob = 0.5
    if p1 in mapping and p2 in mapping:
        idx1 = mapping[p1]
        idx2 = mapping[p2]
        edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(device)
        with torch.no_grad():
            g_out = graph_model(graph_data.x, graph_data.edge_index, edge_label_index)
            graph_prob = torch.sigmoid(g_out).item()
    print("Graph Model Prob:", graph_prob)

    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    if ensemble_path.exists():
        ens = PPIEnsemble(str(ensemble_path))
        bio_comp = bio_mgr.check_localization_compatibility(p1, p2)
        bio_score = bio_comp.get("score", 0.5)
        print("Bio Score:", bio_score)
        
        # We need seq_prob, graph_prob, features
        ens_prob = ens.predict(
            np.array([seq_prob]), 
            np.array([graph_prob]), 
            bio_features=np.array([[bio_score]]),
            method="stacking"
        )[0]
        print("Ensemble Prob:", ens_prob)

if __name__ == "__main__":
    main()
