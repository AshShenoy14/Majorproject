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
from src.data.id_mapper import IDMapper
from src.analysis.explainability import PPIExplainer
from src.analysis.explain_model import explain_prediction as explain_gnn
from src.analysis.network_analysis import NetworkAnalyzer
from src.analysis.mutation_analyzer import MutationAnalyzer
from src.analysis.biological_managers import BiologicalManager
from src.analysis.protein_assistant import ProteinAssistant
from src.utils.bio_encoder import BioFeatureEncoder
from app.backend.schemas import (
    ProteinPair, PredictionResponse, NetworkResponse, BatchPredictionRequest,
    MutationRequest, MutationAnalysisResponse, BioMetaResponse, FeasibilityResponse,
    ResidueGraphRequest, ChatRequest, ChatResponse, IRLMRequest, IRLMResponse, InteractionRegion
)

import torch.nn.functional as F

def insert_novel_node_knn(novel_emb, existing_embs, k=2):
    """
    Computes cosine similarity between a novel protein embedding and existing graph node embeddings,
    returning the keys/IDs of the top-k nearest neighbors.
    """
    if not existing_embs:
        return []
    similarities = {}
    novel_emb_cpu = novel_emb.cpu().float()
    for node_id, emb in existing_embs.items():
        emb_cpu = emb.cpu().float()
        sim = F.cosine_similarity(novel_emb_cpu.unsqueeze(0), emb_cpu.unsqueeze(0)).item()
        similarities[node_id] = sim
    
    # Sort by similarity descending
    sorted_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [node_id for node_id, _ in sorted_neighbors[:k]]

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_system()
    yield

app = FastAPI(
    title="TransGraph-PPI API",
    description="Hybrid Ensemble PPI Prediction System with Real Data",
    lifespan=lifespan
)

# CORS configuration
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

# Global State
models = {}
managers = {}
data_cache = {}
analyzers = {}
explainer = None # Global explainer instance

async def load_system():
    try:
        print("Loading TransGraph-PPI System...")
        
        # 1. Managers
        managers["sequence"] = SequenceManager()
        managers["target"] = TargetManager()
        managers["id_mapper"] = IDMapper()
        managers["bio"] = BiologicalManager()
        managers["bio_encoder"] = BioFeatureEncoder()
        
        # 2. Base Models
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Feature Extractor
        models["esm"] = ESMFeatureExtractor(device=device)
        
        # Sequence Model
        seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
        models["seq_model"] = SequencePPIModel(input_dim=480).to(device)
        if seq_path.exists():
            models["seq_model"].load_state_dict(torch.load(seq_path, map_location=device))
            print("Sequence Model loaded.")
        else:
            print("Warning: Sequence Model weights not found. Using randomly initialized weights.")
        models["seq_model"].eval()

        # Graph Model
        from src.models.graph_model import GATLinkPredictor, GINLinkPredictor
        graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
        graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
        if graph_data_path.exists():
            data_cache["graph"] = torch.load(graph_data_path, weights_only=False).to(device)
            in_channels = data_cache["graph"].x.shape[1]
            
            if graph_path.exists():
                try:
                    state_dict = torch.load(graph_path, map_location=device)
                    # Auto-detect architecture: GIN uses 'convs', SAGEConv uses 'conv1'
                    is_gin = any("convs" in k for k in state_dict.keys())
                    
                    if is_gin:
                        print("Detected GIN architecture for Graph Model.")
                        models["graph_model"] = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
                    else:
                        print("Detected SAGEConv architecture for Graph Model.")
                        models["graph_model"] = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
                    
                    models["graph_model"].load_state_dict(state_dict)
                    print("Graph Model loaded.")
                except Exception as e:
                    print(f"Warning: Could not load Graph Model weights ({e}). Initializing default architecture.")
                    models["graph_model"] = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
            else:
                print("Warning: Graph Model weights not found. Defaulting to SAGEConv.")
                models["graph_model"] = GATLinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
            
            models["graph_model"].eval()
            
            map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
            if map_path.exists():
                 data_cache["mapping"] = torch.load(map_path, weights_only=False)
        else:
            print("Warning: PPI Graph data not found.")

        # 3. Load Ensemble model
        ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
        global explainer # Declare explainer as global to modify it
        if ensemble_path.exists():
            models["ensemble"] = PPIEnsemble(str(ensemble_path))
            print("Loaded Hybrid Ensemble meta-learner.")
            
            # Initialize Explainer
            print("Initializing SHAP explainer...")
            explainer = PPIExplainer(str(ensemble_path))
        else:
            print("Ensemble model not found. Using simple average fallback.")

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

        # 4. Mutation and Novel Analyzers
        if "seq_model" in models and "esm" in models:
            from src.analysis.hotspot_analyzer import HotspotAnalyzer
            from src.analysis.residue_graph_generator import ResidueGraphGenerator
            from src.analysis.irlm_analyzer import IRLMAnalyzer
            
            analyzers["mutation"] = MutationAnalyzer(models["seq_model"], models["esm"], managers["bio"], managers["bio_encoder"])
            analyzers["hotspot"] = HotspotAnalyzer(models["seq_model"], models["esm"], managers["bio"], managers["bio_encoder"])
            analyzers["residue_graph"] = ResidueGraphGenerator(device=device)
            analyzers["irlm"] = IRLMAnalyzer(esm_extractor=models.get("esm"), graph_model=models.get("graph_model"), device=device)
            print("Mutation, Novel, and IRLM Analyzers Ready.")

        # 5. Biological Manager Ready (already initialized above)
        print("Biological Manager Ready (Bio + Cache).")

        # 6. Protein Assistant
        analyzers["assistant"] = ProteinAssistant(
            sequence_manager=managers.get("sequence"),
            target_manager=managers.get("target")
        )
        print("Protein Assistant Ready.")

        print("System Loaded.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FATAL ERROR during startup: {e}")
        # Optionally exit, but for debugging we'll see the print
        sys.exit(1)

@app.post("/predict", 
          response_model=PredictionResponse,
          summary="Predict Interaction probability",
          description="Predicts the interaction probability between two proteins using a hybrid ESM-MLP and GAT ensemble model.",
          tags=["Prediction"])
async def predict_interaction(pair: ProteinPair):
    """
    Computes a hybrid interaction probability for a protein pair.
    
    - **protein1_id**: Identifier for the first protein (e.g., ENSP ID).
    - **protein2_id**: Identifier for the second protein.
    - **protein1_seq** (optional): Amino acid sequence if not in database.
    - **protein2_seq** (optional): Amino acid sequence if not in database.
    
    Returns a unified probability score along with individual model contributions and SHAP explanations.
    """

    try:
        p1 = pair.protein1_id.strip() if pair.protein1_id else None
        p2 = pair.protein2_id.strip() if pair.protein2_id else None
        
        if not p1 or not p2:
             raise HTTPException(status_code=400, detail="Protein IDs are required.")
        
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
            # Add Biological Features to Sequence Input if available
            bio_meta = managers["bio"].get_bio_metadata([p1, p2])
            
            # Helper to get encoded vector for an ID from the fetched metadata
            def get_encoded(pid):
                row = bio_meta[bio_meta["protein_id"] == pid]
                loc_str = row.iloc[0]["localization"] if not row.empty else ""
                return managers["bio_encoder"].encode_protein(loc_str).to(models["esm"].device).float()

            b1 = get_encoded(p1).unsqueeze(0)
            b2 = get_encoded(p2).unsqueeze(0)
            
            e1_final = torch.cat([e1, b1], dim=1)
            e2_final = torch.cat([e2, b2], dim=1)
            
            # Apply sigmoid to raw logits (model outputs logits, not probabilities)
            seq_prob = torch.sigmoid(models["seq_model"](e1_final, e2_final)).item()
            
        # 4. Graph Prediction (with Mapping Resilience and KNN Cold-Start)
        graph_prob = 0.5 
        if "mapping" in data_cache and "graph" in data_cache:
            m_p1 = managers["id_mapper"].resolve_to_graph_id(p1, set(data_cache["mapping"].keys()))
            m_p2 = managers["id_mapper"].resolve_to_graph_id(p2, set(data_cache["mapping"].keys()))
            
            if m_p1 in data_cache["mapping"] and m_p2 in data_cache["mapping"]:
                idx1 = data_cache["mapping"][m_p1]
                idx2 = data_cache["mapping"][m_p2]
            
                edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(models["esm"].device)
                
                with torch.no_grad():
                    g_out = models["graph_model"](data_cache["graph"].x, data_cache["graph"].edge_index, edge_label_index)
                    graph_prob = torch.sigmoid(g_out).item()
            else:
                # One or both proteins are missing from the pre-constructed graph (Cold-Start)
                # Build or retrieve the cached node embeddings dictionary
                if "existing_embeddings" not in data_cache:
                    esm_dim = data_cache["graph"].x.shape[1] - 3
                    data_cache["existing_embeddings"] = {
                        pid: data_cache["graph"].x[idx, :esm_dim].cpu()
                        for pid, idx in data_cache["mapping"].items()
                    }
                
                k = 2  # default top-k nearest neighbors
                
                # Retrieve or find indices for P1
                if m_p1 in data_cache["mapping"]:
                    nb_indices1 = [data_cache["mapping"][m_p1]]
                else:
                    nb_p1 = insert_novel_node_knn(embs[p1], data_cache["existing_embeddings"], k=k)
                    nb_indices1 = [data_cache["mapping"][n] for n in nb_p1]
                
                # Retrieve or find indices for P2
                if m_p2 in data_cache["mapping"]:
                    nb_indices2 = [data_cache["mapping"][m_p2]]
                else:
                    nb_p2 = insert_novel_node_knn(embs[p2], data_cache["existing_embeddings"], k=k)
                    nb_indices2 = [data_cache["mapping"][n] for n in nb_p2]
                
                # Form edge pairs for GAT prediction across all neighbor combinations
                src_indices = []
                dst_indices = []
                for idx1 in nb_indices1:
                    for idx2 in nb_indices2:
                        src_indices.append(idx1)
                        dst_indices.append(idx2)
                
                if src_indices and dst_indices:
                    edge_label_index = torch.tensor([src_indices, dst_indices], dtype=torch.long).to(models["esm"].device)
                    with torch.no_grad():
                        g_out = models["graph_model"](data_cache["graph"].x, data_cache["graph"].edge_index, edge_label_index)
                        graph_prob = torch.sigmoid(g_out).mean().item()
        
        # 5. Final Prediction (Ensemble or Simple Average)
        final_prob = (seq_prob + graph_prob) / 2.0
        model_used = "Average"
        shap_values = None
        bio_match = 0.0 # Default

        if "ensemble" in models and models["ensemble"] is not None and explainer is not None:
            try:
                # Get Biological Match Score
                bio_comp = managers["bio"].check_localization_compatibility(p1, p2)
                bio_score = bio_comp.get("score", 0.5)

                # Enhanced features (7 total): [seq, graph, conf_seq, conf_graph, disagreement, max_conf, bio_score]
                conf_seq = abs(seq_prob - 0.5)
                conf_graph = abs(graph_prob - 0.5)
                disagreement = abs(seq_prob - graph_prob)
                max_conf = max(conf_seq, conf_graph)
                
                ens_prob = models["ensemble"].predict(
                    np.array([seq_prob]), 
                    np.array([graph_prob]), 
                    bio_features=np.array([[bio_score]]),
                    method="stacking"
                )[0]
                
                # --- Zero-Shot Correction (Major Project Polish) ---
                # If both are likely missing from graph (graph_prob is neutral), 
                # but sequence is very strong (ESM > 0.9), and bio-score is high,
                # we should boost the probability to reflect zero-shot confidence.
                if graph_prob == 0.5 and seq_prob > 0.9 and bio_score > 0.8:
                    ens_prob = max(ens_prob, seq_prob * 0.9)
                
                final_prob = float(ens_prob)
                model_used = "XGBoost Ensemble (Zero-Shot Adjusted)"
                
                # Generate SHAP explanation with updated features
                shap_val = explainer.explain_prediction(seq_prob, graph_prob, conf_seq, conf_graph, disagreement, max_conf, bio_score)
                shap_values = shap_val.tolist()[0] 
            except Exception as e:
                print(f"Ensemble prediction/SHAP failed: {e}")
                final_prob = (seq_prob + graph_prob) / 2.0
                model_used = "Average (Ensemble Failed)"
        
        # 6. Explanation
        explanation = {
            "Sequence_Model_Contribution": seq_prob,
            "Graph_Model_Contribution": graph_prob,
            "Model_Used": model_used,
            "SHAP_Values": shap_values,
            "Biological_Match": bio_match > 0.5
        }
        
        # 6a. GNN Topological Explanation (Research Layer with Mapping Resilience)
        gnn_explanation = None
        # Use resolved IDs to ensure topological insights for missing isoforms
        res_p1 = managers["id_mapper"].resolve_to_graph_id(p1, set(data_cache.get("mapping", {}).keys()))
        res_p2 = managers["id_mapper"].resolve_to_graph_id(p2, set(data_cache.get("mapping", {}).keys()))

        if "mapping" in data_cache and res_p1 in data_cache["mapping"] and res_p2 in data_cache["mapping"]:
            try:
                # This call runs GNNExplainer (epochs=50) using resolved IDs
                gnn_explanation = explain_gnn(res_p1, res_p2)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"GNN Explanation failed: {e}")

        # 7. Uniprot ID Mapping for 3D Visuals
        uniprot_maps = {}
        if "id_mapper" in managers:
            uniprot_maps = managers["id_mapper"].ensp_to_uniprot([p1, p2])

        return {
            "interaction_probability": float(final_prob),
            "esm_probability": float(seq_prob),
            "gat_probability": float(graph_prob),
            "confidence_score": abs(float(final_prob) - 0.5) * 2,
            "explanation": explanation,
            "gnn_explanation": gnn_explanation,
            "protein1_uniprot_id": uniprot_maps.get(p1, p1),
            "protein2_uniprot_id": uniprot_maps.get(p2, p2),
            "protein1_seq": sequences[p1],
            "protein2_seq": sequences[p2]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", 
          response_model=List[PredictionResponse],
          summary="Batch Predict Interactions",
          description="Processes multiple protein pairs sequentially for interaction prediction.",
          tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Accepts a list of protein pairs and returns a list of prediction responses.
    """

    results = []
    for pair in request.pairs:
        try:
             res = await predict_interaction(pair)
             results.append(res)
        except Exception as e:
             p1 = pair.protein1_id
             p2 = pair.protein2_id
             print(f"Error in batch for pair {p1}-{p2}: {e}")
             results.append({
                 "interaction_probability": 0.0,
                 "esm_probability": 0.0,
                 "gat_probability": 0.0,
                 "confidence_score": 0.0,
                 "explanation": {
                     "Sequence_Model_Contribution": 0.0,
                     "Graph_Model_Contribution": 0.0,
                     "Model_Used": "Failed",
                     "SHAP_Values": None,
                     "Biological_Match": False
                 },
                 "gnn_explanation": None,
                 "protein1_uniprot_id": p1,
                 "protein2_uniprot_id": p2,
                 "protein1_seq": pair.protein1_seq or "",
                 "protein2_seq": pair.protein2_seq or ""
             })

    return results

@app.get("/network",
         summary="Get Verified Network Subgraph",
         description="Returns a subset of the positive interaction network for visualization.",
         tags=["Analysis"])
async def get_network(limit: int = 100):
    """
    Fetches the top N verified interactions from the training set.
    """

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
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drug_targets",
         summary="Get Drug Targets",
         description="Retrieves known drug targets for a given list of proteins.",
         tags=["Biology"])
async def get_drug_targets(proteins: str = None):
    """
    Returns drug target information from ChEMBL/UniProt mappings.
    """

    try:
        if proteins:
            p_list = proteins.split(",")
        else:
            p_list = ["ENSP00000327694", "ENSP00000373627"] 
            
        df = managers["target"].get_targets(p_list)
        
        if df.empty:
            return []
            
        return df.to_dict(orient="records")
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/centrality",
         summary="Get Network Centrality",
         description="Calculates node centrality metrics for proteins in the interaction network.",
         tags=["Analysis"])
async def get_centrality(top_k: int = 10):
    """
    Returns top-K proteins by degree centrality in the interactome.
    """

    if "network" not in analyzers:
        raise HTTPException(status_code=503, detail="Network Analysis not running (Check train.csv)")
    
    try:
        df = analyzers["network"].calculate_centralities()
        if df.empty:
            return []
        # Return top K nodes by Degree
        return df.head(top_k).to_dict(orient="records")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/stats",
         summary="Get Network Statistics",
         description="Returns global statistics of the protein interaction network.",
         tags=["Analysis"])
async def get_network_stats():
    """
    Returns node count, edge count, and density metrics.
    """

    """
    Get global network statistics.
    """
    if "network" not in analyzers:
        return {}
    return analyzers["network"].get_graph_stats()

@app.post("/analysis/mutate", 
          response_model=MutationAnalysisResponse,
          summary="Analyze Mutation Impact",
          description="Predicts how specific amino acid mutations affect the interaction probability of a protein pair.",
          tags=["Analysis"])
async def scan_mutations(request: MutationRequest):
    """
    Simulates in-silico mutations and measures the delta in interaction probability.
    """

    if "mutation" not in analyzers:
        raise HTTPException(status_code=503, detail="Mutation Analyzer not initialized (Check models)")
    
    try:
        # Get sequences if missing
        sequences = {}
        to_fetch = []
        p1_id, p1_seq = request.protein1_id, request.protein1_seq
        p2_id, p2_seq = request.protein2_id, request.protein2_seq

        if not p1_seq: to_fetch.append(p1_id)
        else: sequences[p1_id] = p1_seq
        
        if not p2_seq: to_fetch.append(p2_id)
        else: sequences[p2_id] = p2_seq

        if to_fetch:
            sequences.update(managers["sequence"].get_sequences(to_fetch))
        
        if p1_id not in sequences or p2_id not in sequences:
             raise HTTPException(status_code=404, detail="Could not find sequences for one or both proteins.")

        results = analyzers["mutation"].project_mutation_impact(
            p1_id, sequences[p1_id],
            p2_id, sequences[p2_id],
            [m.dict() for m in request.mutations]
        )

        if "irlm" in analyzers:
            try:
                graph_data = data_cache.get("graph")
                mapping = data_cache.get("mapping")
                irlm_data = analyzers["irlm"].localize_interaction_regions(
                    p1_id=p1_id,
                    p1_seq=sequences[p1_id],
                    p2_id=p2_id,
                    p2_seq=sequences[p2_id],
                    esm_extractor=models.get("esm"),
                    seq_model=models.get("seq_model"),
                    graph_model=models.get("graph_model"),
                    graph_data=graph_data,
                    mapping=mapping,
                    id_mapper=managers.get("id_mapper")
                )
                results = analyzers["irlm"].annotate_mutations_with_irlm(results, irlm_data)
            except Exception as irlm_err:
                print(f"Warning: IRLM annotation failed during mutation analysis: {irlm_err}")

        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/localize",
          response_model=IRLMResponse,
          summary="Localize Interaction Regions (IRLM)",
          description="Identifies key binding regions and hotspot residues using cross-attention and graph-gated features.",
          tags=["Analysis"])
async def localize_interaction_regions(request: IRLMRequest):
    """
    Computes residue-level cross-attention and graph-gated features to localize interaction regions and hotspots.
    """
    if "irlm" not in analyzers:
        raise HTTPException(status_code=503, detail="IRLM Analyzer not initialized")
    
    try:
        sequences = {}
        to_fetch = []
        p1_id, p1_seq = request.protein1_id, request.protein1_seq
        p2_id, p2_seq = request.protein2_id, request.protein2_seq

        if p1_id and not p1_seq: to_fetch.append(p1_id)
        elif p1_id: sequences[p1_id] = p1_seq

        if p2_id and not p2_seq: to_fetch.append(p2_id)
        elif p2_id: sequences[p2_id] = p2_seq

        if to_fetch:
            sequences.update(managers["sequence"].get_sequences(to_fetch))
        
        p1_seq_final = sequences.get(p1_id, p1_seq)
        p2_seq_final = sequences.get(p2_id, p2_seq)

        if not p1_seq_final or not p2_seq_final:
            raise HTTPException(status_code=400, detail="Protein sequences are required for region localization.")

        graph_data = data_cache.get("graph")
        mapping = data_cache.get("mapping")

        irlm_result = analyzers["irlm"].localize_interaction_regions(
            p1_id=p1_id or "Protein1",
            p1_seq=p1_seq_final,
            p2_id=p2_id or "Protein2",
            p2_seq=p2_seq_final,
            esm_extractor=models.get("esm"),
            seq_model=models.get("seq_model"),
            graph_model=models.get("graph_model"),
            graph_data=graph_data,
            mapping=mapping,
            id_mapper=managers.get("id_mapper"),
            base_probability=request.base_probability
        )

        return IRLMResponse(
            protein1_regions=[InteractionRegion(**r) for r in irlm_result["protein1_regions"]],
            protein2_regions=[InteractionRegion(**r) for r in irlm_result["protein2_regions"]],
            protein1_hotspots=irlm_result["protein1_hotspots"],
            protein2_hotspots=irlm_result["protein2_hotspots"],
            attention_map_shape=irlm_result["attention_map_shape"],
            protein_A_region=irlm_result.get("protein_A_region"),
            protein_B_region=irlm_result.get("protein_B_region"),
            protein_A_importance_scores=irlm_result.get("protein_A_importance_scores"),
            protein_B_importance_scores=irlm_result.get("protein_B_importance_scores"),
            top_residue_pairs=irlm_result.get("top_residue_pairs"),
            region_score=irlm_result.get("region_score", irlm_result.get("region_confidence")),
            region_confidence=irlm_result.get("region_confidence", irlm_result.get("region_score"))
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bio/metadata", 
         response_model=List[BioMetaResponse],
         summary="Get Biological Metadata",
         description="Fetches localization and pathway information for specific proteins.",
         tags=["Biology"])
async def get_bio_metadata(proteins: str):
    """
    Returns subcellular localization and functional pathway data.
    """

    if "bio" not in managers:
        raise HTTPException(status_code=503, detail="Biological Manager not initialized")
    
    try:
        p_list = proteins.split(",")
        df = managers["bio"].get_bio_metadata(p_list)
        return df.to_dict(orient="records")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bio/feasibility", 
         response_model=FeasibilityResponse,
         summary="Check Interaction Feasibility",
         description="Checks if two proteins share compatible subcellular localizations.",
         tags=["Biology"])
async def check_feasibility(p1: str, p2: str):
    """
    Determines if an interaction is physically possible based on biological context.
    """

    if "bio" not in managers:
        raise HTTPException(status_code=503, detail="Biological Manager not initialized")
    
    try:
        return managers["bio"].check_localization_compatibility(p1, p2)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/analysis/hotspots",
          summary="Identify Interaction Hotspots",
          description="Detects critical residues (hotspots) for the interaction using gradient-based importance.",
          tags=["Analysis"])
async def get_hotspots(request: ProteinPair):
    """
    Identifies specific amino acids that contribute most significantly to the interaction.
    """

    if "hotspot" not in analyzers:
        raise HTTPException(status_code=503, detail="Hotspot Analyzer not initialized")
    
    try:
        # Get sequences if missing
        sequences = {}
        to_fetch = []
        if not request.protein1_seq: to_fetch.append(request.protein1_id)
        else: sequences[request.protein1_id] = request.protein1_seq
        if not request.protein2_seq: to_fetch.append(request.protein2_id)
        else: sequences[request.protein2_id] = request.protein2_seq
        
        if to_fetch:
            sequences.update(managers["sequence"].get_sequences(to_fetch))
            
        return analyzers["hotspot"].identify_hotspots(
            request.protein1_id, sequences[request.protein1_id],
            request.protein2_id, sequences[request.protein2_id]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analysis/residue_graph",
          summary="Generate Residue Interaction Graph",
          description="Generates a graph representation of internal residue-residue interactions for a single protein.",
          tags=["Analysis"])
async def get_residue_graph(request: ResidueGraphRequest):
    """
    Constructs a RIG (Residue Interaction Graph) based on distance-truncated contacts.
    """

    if "residue_graph" not in analyzers:
         raise HTTPException(status_code=503, detail="Residue Graph Generator not initialized")
    
    try:
        protein_id = request.protein_id
        sequence = request.sequence
        
        if not sequence:
            seq_dict = managers["sequence"].get_sequences([protein_id])
            if protein_id not in seq_dict:
                raise HTTPException(status_code=404, detail="Sequence not found")
            sequence = seq_dict[protein_id]
            
        uniprot_id = None
        if "id_mapper" in managers:
            uniprot_maps = managers["id_mapper"].ensp_to_uniprot([protein_id])
            uniprot_id = uniprot_maps.get(protein_id)
            
        return analyzers["residue_graph"].generate_rig(sequence, uniprot_id=uniprot_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analysis/optimize",
          summary="Optimize Interaction",
          description="Suggests mutations to either disrupt or enhance a protein-protein interaction.",
          tags=["Analysis"])
async def get_optimization(request: ProteinPair, mode: str = "disrupt"):
    """
    In-silico optimization or disruption of a PPI.
    """

    if "mutation" not in analyzers:
        raise HTTPException(status_code=503, detail="Mutation Analyzer not initialized")
    
    try:
        # Get sequences
        sequences = {}
        to_fetch = []
        if not request.protein1_seq: to_fetch.append(request.protein1_id)
        else: sequences[request.protein1_id] = request.protein1_seq
        if not request.protein2_seq: to_fetch.append(request.protein2_id)
        else: sequences[request.protein2_id] = request.protein2_seq
        
        if to_fetch:
            sequences.update(managers["sequence"].get_sequences(to_fetch))
            
        return analyzers["mutation"].suggest_optimal_mutations(
            request.protein1_id, sequences[request.protein1_id],
            request.protein2_id, sequences[request.protein2_id],
            mode=mode
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/vulnerability",
         summary="Calculate Pathway Vulnerability",
         description="Assesses how fragile a biological pathway is to mutations in a specific protein pair.",
         tags=["Analysis"])
async def get_vulnerability(p1: str, p2: str, delta: float):
    """
    Calculates vulnerability scores for downstream pathways.
    """

    if "bio" not in managers:
        raise HTTPException(status_code=503, detail="Biological Manager not initialized")
    return managers["bio"].calculate_pathway_vulnerability(p1, p2, delta)

@app.get("/chat/greeting",
         summary="Get AI Assistant Greeting",
         description="Returns a welcome message and suggested questions for the protein assistant.",
         tags=["AI Assistant"])
async def get_chat_greeting():
    if "assistant" not in analyzers:
        raise HTTPException(status_code=503, detail="Protein Assistant not initialized")
    return analyzers["assistant"].get_greeting()


@app.post("/chat",
          response_model=ChatResponse,
          summary="Chat with Protein AI Assistant",
          description="Ask questions about proteins, diseases, drug targets, and biology concepts.",
          tags=["AI Assistant"])
async def chat_with_assistant(request: ChatRequest):
    if "assistant" not in analyzers:
        raise HTTPException(status_code=503, detail="Protein Assistant not initialized")
    
    try:
        result = analyzers["assistant"].answer(request.message)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Auto-reload trigger
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

