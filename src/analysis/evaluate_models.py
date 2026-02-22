import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import sys
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def evaluate_models(embedding_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on {device}...")
    
    # 1. Load Data
    print("Loading Data...")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv") # Using test set if available, else val
    embeddings = torch.load(embedding_path, weights_only=False)
    graph_data = torch.load(PROCESSED_DATA_DIR / "ppi_graph.pt", weights_only=False).to(device)
    graph_mapping = torch.load(PROCESSED_DATA_DIR / "ppi_graph_mapping.pt", weights_only=False)
    
    # 2. Load Models
    print("Loading Models...")
    
    # Baseline RF
    rf_path = PROJECT_ROOT / "models" / "baseline_rf.pkl"
    rf_model = joblib.load(rf_path) if rf_path.exists() else None
    
    # Sequence Model
    seq_model = SequencePPIModel(input_dim=320).to(device)
    seq_model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "sequence_model_best.pth", map_location=device))
    seq_model.eval()
    
    # Graph Model
    in_channels = graph_data.x.shape[1]
    gat_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=64).to(device)
    gat_model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "graph_model_best.pth", map_location=device))
    gat_model.eval()
    
    # Ensemble
    ensemble = PPIEnsemble(meta_model_path=str(PROJECT_ROOT / "models" / "ensemble_model.pkl"))
    
    # 3. Evaluation Loop
    results = {
        "y_true": [],
        "rf_prob": [],
        "seq_prob": [],
        "gat_prob": [],
        "ens_prob": []
    }
    
    print("Running Predictions...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        
        if p1 not in embeddings or p2 not in embeddings:
            continue
            
        results["y_true"].append(label)
        
        # RF Prediction
        if rf_model:
            feat = np.concatenate([embeddings[p1].numpy(), embeddings[p2].numpy()]).reshape(1, -1)
            results["rf_prob"].append(rf_model.predict_proba(feat)[0, 1])
        else:
             results["rf_prob"].append(0.5)

        # Sequence Prediction
        e1 = embeddings[p1].unsqueeze(0).to(device)
        e2 = embeddings[p2].unsqueeze(0).to(device)
        with torch.no_grad():
            s_prob = seq_model(e1, e2).item()
            results["seq_prob"].append(s_prob)
            
            # GAT Prediction
            g_prob = 0.5
            if p1 in graph_mapping and p2 in graph_mapping:
                idx1 = graph_mapping[p1]
                idx2 = graph_mapping[p2]
                edge_idx = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(device)
                g_prob = gat_model(graph_data.x, graph_data.edge_index, edge_idx).item()
            results["gat_prob"].append(g_prob)
            
            # Ensemble
            e_prob = ensemble.predict(np.array([s_prob]), np.array([g_prob]), method="stacking" if ensemble.meta_model else "soft_voting")[0]
            results["ens_prob"].append(e_prob)

    # 4. Calculate Metrics
    y_true = np.array(results["y_true"])
    metrics = []
    
    models_to_eval = [("ESM-2 (Seq)", results["seq_prob"]), 
                      ("GAT (Graph)", results["gat_prob"]), 
                      ("Hybrid Ensemble", results["ens_prob"])]
    
    if rf_model:
        models_to_eval.insert(0, ("Random Forest", results["rf_prob"]))
        
    print("\n" + "="*40)
    print(f"{'Model':<20} | {'Acc':<8} | {'AUC':<8} | {'F1':<8}")
    print("-" * 40)
    
    for name, probs in models_to_eval:
        probs = np.array(probs)
        preds = (probs > 0.5).astype(int)
        
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, probs)
        f1 = f1_score(y_true, preds)
        
        print(f"{name:<20} | {acc:.4f}   | {auc:.4f}   | {f1:.4f}")
        
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_models(args.embedding_path)
