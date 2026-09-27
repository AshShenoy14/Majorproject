"""
Isolated GAT Comparison Backend for TransGraph-PPI.
Port: 8001
Role: Controlled Experimental Comparison (Original Proposal Architecture)

Loads ONLY frozen Standard GAT artifacts:
- ESMGAT/weights/gat_model_best.pth
- ESMGAT/weights/gat_calibrator.json
- ESMGAT/weights/gat_ensemble_model.pkl

Completely isolated from the primary GraphSAGE production backend on Port 8000.
"""
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.paths import PROCESSED_DATA_DIR
from src.utils.esm_config import ESM_EMBED_DIM
from src.models.sequence_model import SequencePPIModel
from src.data.feature_extraction import ESMFeatureExtractor
from src.data.sequence_manager import SequenceManager
from src.data.id_mapper import IDMapper
from ESMGAT.models.gat_model import GATLinkPredictor
from ESMGAT.models.gat_ensemble import GATPPIEnsemble, GAT_META_FEATURE_NAMES


# Schemas
class ProteinPair(BaseModel):
    protein1_id: Optional[str] = Field(None, description="Identifier for protein 1 (e.g., ENSP ID)")
    protein2_id: Optional[str] = Field(None, description="Identifier for protein 2 (e.g., ENSP ID)")
    protein1_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 1")
    protein2_seq: Optional[str] = Field(None, description="Amino acid sequence for protein 2")


class BatchPredictionRequest(BaseModel):
    pairs: List[ProteinPair] = Field(..., description="List of protein pairs to predict")


# State container
gat_state = {
    "device": "cpu",
    "seq_manager": None,
    "id_mapper": None,
    "feature_extractor": None,
    "seq_model": None,
    "gat_model": None,
    "gat_ensemble": None,
    "shap_explainer": None,
    "graph": None,
    "mapping": None,
    "existing_embeddings": None,
}


def load_gat_system():
    print("[GAT Backend] Initializing Controlled Comparison Backend on Port 8001...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gat_state["device"] = device
    print(f"[GAT Backend] Using compute device: {device}")

    # 1. Managers
    gat_state["seq_manager"] = SequenceManager()
    gat_state["id_mapper"] = IDMapper()

    # 2. ESM Feature Extractor
    gat_state["feature_extractor"] = ESMFeatureExtractor(device=device)

    # 3. Shared Sequence Model
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    if not seq_path.exists():
        raise RuntimeError(f"Sequence model missing at {seq_path}")
    seq_model = SequencePPIModel(input_dim=ESM_EMBED_DIM).to(device)
    seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()
    gat_state["seq_model"] = seq_model
    print("[GAT Backend] Sequence Model loaded successfully.")

    # 4. Graph Data
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    if not graph_data_path.exists() or not map_path.exists():
        raise RuntimeError(f"Graph data or mapping missing in {PROCESSED_DATA_DIR}")

    graph = torch.load(graph_data_path, weights_only=False).to(device)
    gat_state["graph"] = graph
    gat_state["mapping"] = torch.load(map_path, weights_only=False)
    in_channels = graph.x.shape[1]
    print(f"[GAT Backend] Graph loaded with {graph.num_nodes} nodes, in_channels={in_channels}.")

    # 5. Standard GAT Model (Frozen comparison model)
    gat_weights_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_model_best.pth"
    if not gat_weights_path.exists():
        raise RuntimeError(f"Standard GAT weights missing at {gat_weights_path}")

    gat_model = GATLinkPredictor(in_channels=in_channels, hidden_channels=256, heads=4).to(device)
    gat_model.load_state_dict(torch.load(gat_weights_path, map_location=device))
    gat_model.eval()
    gat_state["gat_model"] = gat_model
    print(f"[GAT Backend] Standard GAT Model loaded successfully from {gat_weights_path}.")

    # 6. GAT XGBoost Ensemble & Calibrator
    ensemble_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_ensemble_model.pkl"
    calibrator_path = PROJECT_ROOT / "ESMGAT" / "weights" / "gat_calibrator.json"
    if not ensemble_path.exists() or not calibrator_path.exists():
        raise RuntimeError(f"GAT ensemble ({ensemble_path}) or calibrator ({calibrator_path}) missing.")

    gat_ensemble = GATPPIEnsemble(meta_model_path=str(ensemble_path), calibrator_path=str(calibrator_path))
    gat_state["gat_ensemble"] = gat_ensemble
    print("[GAT Backend] GAT Stacking Ensemble loaded successfully.")

    # 7. SHAP Explainer
    try:
        gat_state["shap_explainer"] = shap.TreeExplainer(gat_ensemble.meta_model)
        print("[GAT Backend] GAT SHAP TreeExplainer initialized successfully.")
    except Exception as e:
        print(f"[GAT Backend] Warning: SHAP initialization failed: {e}")
        gat_state["shap_explainer"] = None

    print("[GAT Backend] Initialization complete. Ready to serve comparison queries.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_gat_system()
    yield


app = FastAPI(
    title="TransGraph-PPI GAT Comparison Backend",
    description="Controlled Comparison Service serving Standard GAT (Original Proposal Architecture)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "TransGraph-PPI Controlled Comparison Backend",
        "model": "ESM-2 + Standard GAT + XGBoost",
        "architecture": "Standard GAT (Original Proposal)",
        "role": "Controlled Experimental Comparison",
        "port": 8001,
        "threshold": 0.50,
        "status_code": 200
    }


@app.get("/info")
def model_info():
    return {
        "model_name": "ESM-2 + Standard GAT + XGBoost",
        "role": "Controlled Experimental Comparison",
        "architecture": "Standard GAT (4-head self-attention)",
        "metrics": {
            "accuracy": 0.8979,
            "precision": 0.9205,
            "recall": 0.8710,
            "f1_score": 0.8951,
            "roc_auc": 0.9582,
            "pr_auc": 0.9627
        },
        "threshold": 0.50,
        "features": GAT_META_FEATURE_NAMES
    }


def _run_single_gat_prediction(p1: str, p2: str, s1: Optional[str] = None, s2: Optional[str] = None) -> Dict[str, Any]:
    # 1. Resolve sequences
    sequences = {}
    to_fetch = []
    if s1: sequences[p1] = s1
    else: to_fetch.append(p1)

    if s2: sequences[p2] = s2
    else: to_fetch.append(p2)

    if to_fetch:
        fetched = gat_state["seq_manager"].get_sequences(to_fetch)
        sequences.update(fetched)

    if p1 not in sequences or p2 not in sequences:
        raise HTTPException(status_code=404, detail=f"Could not find amino acid sequences for '{p1}' and/or '{p2}'.")

    # 2. Compute ESM embeddings
    embs = gat_state["feature_extractor"].get_embeddings(sequences, batch_size=2)
    device = gat_state["device"]
    e1 = embs[p1].unsqueeze(0).to(device).float()
    e2 = embs[p2].unsqueeze(0).to(device).float()

    # 3. Sequence branch prediction
    with torch.no_grad():
        seq_prob = torch.sigmoid(gat_state["seq_model"](e1, e2)).item()

    # 4. GAT Graph branch prediction
    graph_prob = 0.5
    mapping = gat_state["mapping"]
    graph = gat_state["graph"]

    m_p1 = gat_state["id_mapper"].resolve_to_graph_id(p1, set(mapping.keys()))
    m_p2 = gat_state["id_mapper"].resolve_to_graph_id(p2, set(mapping.keys()))

    if m_p1 in mapping and m_p2 in mapping:
        idx1 = mapping[m_p1]
        idx2 = mapping[m_p2]
        edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long, device=device)
        with torch.no_grad():
            g_out = gat_state["gat_model"](graph.x, graph.edge_index, edge_label_index)
            graph_prob = torch.sigmoid(g_out).item()
    else:
        # Novel / missing node fallback using top-k cosine similarity
        if gat_state["existing_embeddings"] is None:
            esm_dim = graph.x.shape[1] - 3
            gat_state["existing_embeddings"] = {
                pid: graph.x[idx, :esm_dim].cpu() for pid, idx in mapping.items()
            }
        existing_embs = gat_state["existing_embeddings"]

        def get_knn(emb, k=2):
            sims = {}
            emb_cpu = emb.cpu().float()
            for node_id, e_node in existing_embs.items():
                sims[node_id] = F.cosine_similarity(emb_cpu.unsqueeze(0), e_node.cpu().float().unsqueeze(0)).item()
            return [node_id for node_id, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:k]]

        nb1 = [mapping[m_p1]] if m_p1 in mapping else [mapping[n] for n in get_knn(embs[p1])]
        nb2 = [mapping[m_p2]] if m_p2 in mapping else [mapping[n] for n in get_knn(embs[p2])]

        src_indices = [i1 for i1 in nb1 for i2 in nb2]
        dst_indices = [i2 for i1 in nb1 for i2 in nb2]
        if src_indices and dst_indices:
            edge_label_index = torch.tensor([src_indices, dst_indices], dtype=torch.long, device=device)
            with torch.no_grad():
                g_out = gat_state["gat_model"](graph.x, graph.edge_index, edge_label_index)
                graph_prob = torch.sigmoid(g_out).mean().item()

    # 5. GAT Stacking Ensemble Prediction
    gat_ensemble = gat_state["gat_ensemble"]
    gat_prob_cal = float(gat_ensemble.calibrate_gat(np.array([graph_prob]))[0])
    conf_seq = abs(seq_prob - 0.5)
    conf_gat = abs(gat_prob_cal - 0.5)
    diff = abs(seq_prob - gat_prob_cal)
    max_conf = max(conf_seq, conf_gat)

    ens_prob = float(gat_ensemble.predict_proba(np.array([seq_prob]), np.array([graph_prob]))[0])
    threshold = 0.50  # Frozen validation-derived threshold for GAT Ensemble

    # 6. SHAP Attribution
    shap_values = [0.0] * 7
    if gat_state["shap_explainer"] is not None:
        try:
            consensus = seq_prob * gat_prob_cal
            X_meta = np.array([[seq_prob, gat_prob_cal, conf_seq, conf_gat, diff, max_conf, consensus]])
            sv = gat_state["shap_explainer"].shap_values(X_meta)
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            shap_values = sv[0].tolist()
        except Exception as e:
            print(f"[GAT Backend] SHAP calculation note: {e}")

    # 7. UniProt mapping
    uniprot_maps = gat_state["id_mapper"].ensp_to_uniprot([p1, p2])

    pred_label = "Likely Interaction" if ens_prob >= threshold else "Unlikely Interaction"

    return {
        "protein1_id": p1,
        "protein2_id": p2,
        "status": "success",
        "error": None,
        "interaction_probability": float(ens_prob),
        "esm_probability": float(seq_prob),
        "gat_probability": float(graph_prob),
        "confidence_score": abs(float(ens_prob) - 0.5) * 2,
        "model_name": "ESM-2 + Standard GAT + XGBoost",
        "role": "Controlled Experimental Comparison",
        "threshold": threshold,
        "prediction_label": pred_label,
        "explanation": {
            "Sequence_Model_Contribution": float(seq_prob),
            "Graph_Model_Contribution": float(graph_prob),
            "Model_Used": "ESM-2 + Standard GAT + XGBoost",
            "SHAP_Values": shap_values,
            "SHAP_Sequence": shap_values[0] if shap_values else 0.0,
            "SHAP_Graph": shap_values[1] if len(shap_values) > 1 else 0.0,
            "Role": "Controlled Experimental Comparison"
        },
        "shap_explanations": shap_values,
        "gnn_explanation": None,
        "protein1_uniprot_id": uniprot_maps.get(p1, p1),
        "protein2_uniprot_id": uniprot_maps.get(p2, p2),
        "protein1_seq": sequences.get(p1),
        "protein2_seq": sequences.get(p2)
    }


@app.post("/predict")
def predict_interaction(pair: ProteinPair):
    p1 = pair.protein1_id.strip() if pair.protein1_id else None
    p2 = pair.protein2_id.strip() if pair.protein2_id else None
    if not p1 or not p2:
        raise HTTPException(status_code=400, detail="Protein IDs are required.")

    return _run_single_gat_prediction(p1, p2, pair.protein1_seq, pair.protein2_seq)


@app.post("/predict_batch")
def predict_batch(request: BatchPredictionRequest):
    results = []
    for pair in request.pairs:
        p1 = pair.protein1_id.strip() if pair.protein1_id else None
        p2 = pair.protein2_id.strip() if pair.protein2_id else None
        if not p1 or not p2:
            results.append({"status": "error", "error": "Missing protein ID"})
            continue
        try:
            res = _run_single_gat_prediction(p1, p2, pair.protein1_seq, pair.protein2_seq)
            results.append(res)
        except Exception as e:
            results.append({"status": "error", "error": str(e), "protein1_id": p1, "protein2_id": p2})
    return results


if __name__ == "__main__":
    uvicorn.run("ESMGAT.backend.main:app", host="0.0.0.0", port=8001, reload=False)
