import sys
import os
from pathlib import Path

# Add project root and configure environments BEFORE importing heavy ML libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
import pandas as pd
import joblib
from typing import List

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.data.feature_extraction import ESMFeatureExtractor
from src.data.sequence_manager import SequenceManager
from src.data.target_manager import TargetManager
from src.analysis.explainability import PPIExplainer
from src.analysis.network_analysis import NetworkAnalyzer
from app.backend.schemas import ProteinPair, PredictionResponse, NetworkResponse, BatchPredictionRequest

app = FastAPI(title="TransGraph-PPI API", description="Hybrid Ensemble PPI Prediction System with Real Data")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
models = {}
managers = {}
data_cache = {}
analyzers = {}

@app.on_event("startup")
async def load_system():
    print("Loading TransGraph-PPI System...")
    
    # 1. Managers
    managers["sequence"] = SequenceManager()
    managers["target"] = TargetManager()
    
    # 2. Base Models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Feature Extractor
    models["esm"] = ESMFeatureExtractor(device=device)
    
    # Sequence Model
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    models["seq_model"] = SequencePPIModel(input_dim=320).to(device)
    if seq_path.exists():
        models["seq_model"].load_state_dict(torch.load(seq_path, map_location=device))
        models["seq_model"].eval()
        print("Sequence Model loaded.")
    else:
        print("Warning: Sequence Model weights not found.")

    # Graph Model
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    if graph_data_path.exists():
        data_cache["graph"] = torch.load(graph_data_path, weights_only=False).to(device)
        in_channels = data_cache["graph"].x.shape[1]
        models["graph_model"] = GATLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
        if graph_path.exists():
            try:
                models["graph_model"].load_state_dict(torch.load(graph_path, map_location=device))
                models["graph_model"].eval()
                print("Graph Model loaded.")
            except Exception as e:
                print(f"Warning: Could not load Graph Model weights ({e}). Graph predictions will use defaults.")
        else:
            print("Warning: Graph Model weights not found.")
        
        map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
        if map_path.exists():
             data_cache["mapping"] = torch.load(map_path, weights_only=False)
    else:
        print("Warning: PPI Graph data not found.")

    # Ensemble
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    models["ensemble"] = PPIEnsemble(meta_model_path=str(ensemble_path) if ensemble_path.exists() else None)
    
    # Explainer
    if models["ensemble"].meta_model:
        models["explainer"] = PPIExplainer(meta_model_path=str(ensemble_path))

    # Network Analyzer
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if train_path.exists():
        print("Initializing Network Analyzer...")
        df = pd.read_csv(train_path)
        # Filter only positive interactions for analysis graph
        df_pos = df[df['label'] == 1]
        analyzers["network"] = NetworkAnalyzer()
        analyzers["network"].build_from_dataframe(df_pos)
        print("Network Analyzer Ready.")

    print("System Loaded.")

@app.post("/predict", response_model=PredictionResponse)
async def predict_interaction(pair: ProteinPair):
    try:
        p1, p2 = pair.protein1_id, pair.protein2_id
        
        # 1. Get Sequences
        sequences = {}
        to_fetch = []
        
        if not pair.protein1_seq: to_fetch.append(p1)
        else: sequences[p1] = pair.protein1_seq
            
        if not pair.protein2_seq: to_fetch.append(p2)
        else: sequences[p2] = pair.protein2_seq
        
        if to_fetch:
            fetched = managers["sequence"].get_sequences(to_fetch)
            sequences.update(fetched)
            
        if p1 not in sequences or p2 not in sequences:
             raise HTTPException(status_code=404, detail="Could not find sequences for one or both proteins.")

        # 2. Get Embeddings
        embs = models["esm"].get_embeddings(sequences, batch_size=2)
        e1 = embs[p1].unsqueeze(0).to(models["esm"].device).float()
        e2 = embs[p2].unsqueeze(0).to(models["esm"].device).float()
        
        # 3. Sequence Prediction
        with torch.no_grad():
            # Apply sigmoid to raw logits (model outputs logits, not probabilities)
            seq_prob = torch.sigmoid(models["seq_model"](e1, e2)).item()
            
        # 4. Graph Prediction
        graph_prob = 0.5 
        if "mapping" in data_cache and p1 in data_cache["mapping"] and p2 in data_cache["mapping"]:
            idx1 = data_cache["mapping"][p1]
            idx2 = data_cache["mapping"][p2]
            
            edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(models["esm"].device)
            
            with torch.no_grad():
                g_out = models["graph_model"](data_cache["graph"].x, data_cache["graph"].edge_index, edge_label_index)
                graph_prob = torch.sigmoid(g_out).item()
        
        # 5. Ensemble Prediction
        if models["ensemble"].meta_model:
            # Replicate PPIEnsemble._build_features logic for the single sample
            conf_seq = abs(seq_prob - 0.5)
            conf_graph = abs(graph_prob - 0.5)
            X_input = np.array([[seq_prob, graph_prob, conf_seq, conf_graph]])
            final_prob = models["ensemble"].meta_model.predict_proba(X_input)[0, 1]
        else:
            final_prob = (seq_prob + graph_prob) / 2
        
        # 6. Explanation
        explanation = {
            "Sequence_Model_Contribution": seq_prob,
            "Graph_Model_Contribution": graph_prob,
            "Sequence_Confidence": conf_seq,
            "Graph_Confidence": conf_graph,
        }
        
        if "explainer" in models:
            shap_vals = models["explainer"].explain_prediction(seq_prob, graph_prob, conf_seq, conf_graph)
            # shap_vals[0] will have 4 elements corresponding to the 4 features
            explanation["SHAP_Sequence"] = float(shap_vals[0][0])
            explanation["SHAP_Graph"] = float(shap_vals[0][1])
            explanation["SHAP_Seq_Conf"] = float(shap_vals[0][2])
            explanation["SHAP_Graph_Conf"] = float(shap_vals[0][3])

        return {
            "interaction_probability": float(final_prob),
            "esm_probability": float(seq_prob),
            "gat_probability": float(graph_prob),
            "confidence_score": abs(float(final_prob) - 0.5) * 2,
            "explanation": explanation
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    results = []
    for pair in request.pairs:
        try:
             res = await predict_interaction(pair)
             results.append(res)
        except Exception as e:
             # Log the error but continue or fail? We'll fail the batch if one severely errors for now, or just raise.
             # Ideally we return a partial result, but for simplicity we raise 500
             raise HTTPException(status_code=500, detail=f"Error processing pair {pair.protein1_id}-{pair.protein2_id}: {str(e)}")
    return results

@app.get("/network")
async def get_network(limit: int = 100):
    try:
        train_path = PROCESSED_DATA_DIR / "train.csv"
        if not train_path.exists():
             return {"nodes": [], "edges": []}
             
        df = pd.read_csv(train_path)
        df = df[df["label"] == 1].head(limit)
        
        nodes = set()
        edges = []
        
        for _, row in df.iterrows():
            p1, p2 = row["protein1"], row["protein2"]
            nodes.add(p1)
            nodes.add(p2)
            edges.append({"source": p1, "target": p2, "weight": 1.0, "type": "verified"})
            
        node_list = [{"id": n, "label": n} for n in nodes]
        
        return {"nodes": node_list, "edges": edges}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drug_targets")
async def get_drug_targets(proteins: str = None):
    try:
        if proteins:
            p_list = proteins.split(",")
        else:
            p_list = ["ENSP00000327694", "ENSP00000373627"] 
            
        df = managers["target"].get_targets(p_list)
        
        if df.empty:
            return []
            
        return df.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/centrality")
async def get_centrality(top_k: int = 10):
    """
    Get top centrality metrics for nodes in the network.
    """
    if "network" not in analyzers:
        raise HTTPException(status_code=503, detail="Network Analysis not running (Check train.csv)")
    
    try:
        df = analyzers["network"].calculate_centralities()
        if df.empty:
            return []
        # Return top K nodes by Degree
        return df.head(top_k).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/stats")
async def get_network_stats():
    """
    Get global network statistics.
    """
    if "network" not in analyzers:
        return {}
    return analyzers["network"].get_graph_stats()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
