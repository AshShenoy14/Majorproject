from typing import List

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from app.backend import state
from app.backend.schemas import PredictionResponse, ProteinPair, BatchPredictionRequest

router = APIRouter(tags=["Prediction"])


@router.post("/predict",
             response_model=PredictionResponse,
             summary="Predict Interaction probability",
             description="Predicts the interaction probability between two proteins using a hybrid ESM-MLP and GraphSAGE ensemble model.")
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
            fetched = state.managers["sequence"].get_sequences(to_fetch)
            sequences.update(fetched)

        if p1 not in sequences or p2 not in sequences:
            raise HTTPException(status_code=404, detail="Could not find sequences for one or both proteins.")

        # 2. Get Embeddings
        embs = state.models["esm"].get_embeddings(sequences, batch_size=2)
        e1 = embs[p1].unsqueeze(0).to(state.models["esm"].device).float()
        e2 = embs[p2].unsqueeze(0).to(state.models["esm"].device).float()

        # 3. Sequence Prediction
        with torch.no_grad():
            # Apply sigmoid to raw logits (model outputs logits, not probabilities)
            seq_prob = torch.sigmoid(state.models["seq_model"](e1, e2)).item()

        # 4. Graph Prediction (with Mapping Resilience and KNN Cold-Start)
        graph_prob = 0.5
        if "mapping" in state.data_cache and "graph" in state.data_cache:
            m_p1 = state.managers["id_mapper"].resolve_to_graph_id(p1, set(state.data_cache["mapping"].keys()))
            m_p2 = state.managers["id_mapper"].resolve_to_graph_id(p2, set(state.data_cache["mapping"].keys()))

            if m_p1 in state.data_cache["mapping"] and m_p2 in state.data_cache["mapping"]:
                idx1 = state.data_cache["mapping"][m_p1]
                idx2 = state.data_cache["mapping"][m_p2]

                edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(state.models["esm"].device)

                with torch.no_grad():
                    g_out = state.models["graph_model"](state.data_cache["graph"].x, state.data_cache["graph"].edge_index, edge_label_index)
                    graph_prob = torch.sigmoid(g_out).item()
            else:
                # One or both proteins are missing from the pre-constructed graph (Cold-Start)
                # Build or retrieve the cached node embeddings dictionary
                if "existing_embeddings" not in state.data_cache:
                    esm_dim = state.data_cache["graph"].x.shape[1] - 3
                    state.data_cache["existing_embeddings"] = {
                        pid: state.data_cache["graph"].x[idx, :esm_dim].cpu()
                        for pid, idx in state.data_cache["mapping"].items()
                    }

                k = 2  # default top-k nearest neighbors

                # Retrieve or find indices for P1
                if m_p1 in state.data_cache["mapping"]:
                    nb_indices1 = [state.data_cache["mapping"][m_p1]]
                else:
                    nb_p1 = state.insert_novel_node_knn(embs[p1], state.data_cache["existing_embeddings"], k=k)
                    nb_indices1 = [state.data_cache["mapping"][n] for n in nb_p1]

                # Retrieve or find indices for P2
                if m_p2 in state.data_cache["mapping"]:
                    nb_indices2 = [state.data_cache["mapping"][m_p2]]
                else:
                    nb_p2 = state.insert_novel_node_knn(embs[p2], state.data_cache["existing_embeddings"], k=k)
                    nb_indices2 = [state.data_cache["mapping"][n] for n in nb_p2]

                # Form edge pairs for GraphSAGE prediction across all neighbor combinations
                src_indices = []
                dst_indices = []
                for idx1 in nb_indices1:
                    for idx2 in nb_indices2:
                        src_indices.append(idx1)
                        dst_indices.append(idx2)

                if src_indices and dst_indices:
                    edge_label_index = torch.tensor([src_indices, dst_indices], dtype=torch.long).to(state.models["esm"].device)
                    with torch.no_grad():
                        g_out = state.models["graph_model"](state.data_cache["graph"].x, state.data_cache["graph"].edge_index, edge_label_index)
                        graph_prob = torch.sigmoid(g_out).mean().item()

        # 5. Final Prediction (Strict XGBoost Meta-Learner Ensemble)
        if "ensemble" not in state.models or state.models["ensemble"] is None or state.explainer is None:
            raise HTTPException(status_code=503, detail="Ensemble meta-learner or SHAP explainer unavailable.")

        try:
            # Display-only: live UniProt-backed compatibility, NOT fed to the ensemble
            bio_comp = state.managers["bio"].check_localization_compatibility(p1, p2, persist=False)
            bio_match = bio_comp.get("score", 0.5)

            # The meta-learner consumes the CALIBRATED GraphSAGE probability
            graph_prob_cal = float(state.models["ensemble"].calibrate_graph(np.array([graph_prob]))[0])
            conf_seq = abs(seq_prob - 0.5)
            conf_graph = abs(graph_prob_cal - 0.5)
            disagreement = abs(seq_prob - graph_prob_cal)
            max_conf = max(conf_seq, conf_graph)

            # predict() takes the raw graph probability and calibrates it internally.
            # 7 meta-features: [p_seq, p_graph, conf_seq, conf_graph, diff, max_conf, consensus]
            ens_prob = state.models["ensemble"].predict(
                np.array([seq_prob]),
                np.array([graph_prob]),
                method="stacking"
            )[0]

            feat_matrix = state.models["ensemble"]._build_features(np.array([seq_prob]), np.array([graph_prob_cal]))
            assert feat_matrix.shape[1] == 7, f"Feature parity failure: Expected 7 features for XGBoost ensemble, got {feat_matrix.shape[1]}"

            final_prob = float(ens_prob)
            model_used = "XGBoost Ensemble"

            # Generate SHAP explanation with 7 features
            shap_val = state.explainer.explain_prediction(seq_prob, graph_prob_cal, conf_seq, conf_graph, disagreement, max_conf)
            shap_values = shap_val.tolist()[0]
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Ensemble prediction / SHAP explanation failed: {e}")

        # 6. Explanation
        explanation = {
            "Sequence_Model_Contribution": seq_prob,
            "Graph_Model_Contribution": graph_prob,
            "Model_Used": model_used,
            "SHAP_Values": shap_values,
            "SHAP_Sequence": shap_values[0] if shap_values and len(shap_values) > 0 else 0.0,
            "SHAP_Graph": shap_values[1] if shap_values and len(shap_values) > 1 else 0.0,
            "Biological_Match": bio_match > 0.5
        }

        # 6a. GNN Topological Evidence (< 2ms fast inspection from in-memory graph)
        gnn_explanation = None
        # Use resolved IDs to ensure topological insights for missing isoforms
        res_p1 = state.managers["id_mapper"].resolve_to_graph_id(p1, set(state.data_cache.get("mapping", {}).keys()))
        res_p2 = state.managers["id_mapper"].resolve_to_graph_id(p2, set(state.data_cache.get("mapping", {}).keys()))

        if "mapping" in state.data_cache and res_p1 in state.data_cache["mapping"] and res_p2 in state.data_cache["mapping"]:
            try:
                from src.analysis.explain_model import get_topological_neighbors
                gnn_explanation = get_topological_neighbors(
                    res_p1, res_p2,
                    data=state.data_cache.get("graph"),
                    node_mapping=state.data_cache.get("mapping")
                )
            except Exception as e:
                print(f"Topological graph inspection failed: {e}")

        # 7. Uniprot ID Mapping for 3D Visuals
        uniprot_maps = {}
        if "id_mapper" in state.managers:
            uniprot_maps = state.managers["id_mapper"].ensp_to_uniprot([p1, p2])

        return {
            "protein1_id": p1,
            "protein2_id": p2,
            "status": "success",
            "error": None,
            "interaction_probability": float(final_prob),
            "esm_probability": float(seq_prob),
            "gat_probability": float(graph_prob),
            "confidence_score": abs(float(final_prob) - 0.5) * 2,
            "explanation": explanation,
            "shap_explanations": shap_values,
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


@router.post("/predict_batch",
             response_model=List[PredictionResponse],
             summary="Batch Predict Interactions",
             description="Processes multiple protein pairs sequentially for interaction prediction.")
async def predict_batch(request: BatchPredictionRequest):
    """
    Accepts a list of protein pairs and returns a list of prediction responses.
    """

    results = []
    for pair in request.pairs:
        p1 = pair.protein1_id or "Unknown_P1"
        p2 = pair.protein2_id or "Unknown_P2"
        try:
            res = await predict_interaction(pair)
            results.append(res)
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, "detail"):
                error_msg = str(e.detail)
            print(f"Error in batch for pair {p1}-{p2}: {error_msg}")
            results.append({
                "protein1_id": p1,
                "protein2_id": p2,
                "status": "error",
                "error": error_msg,
                "interaction_probability": None,
                "esm_probability": None,
                "gat_probability": None,
                "confidence_score": None,
                "explanation": {
                    "Sequence_Model_Contribution": 0.0,
                    "Graph_Model_Contribution": 0.0,
                    "Model_Used": "Failed",
                    "SHAP_Values": None,
                    "Biological_Match": False
                },
                "shap_explanations": None,
                "gnn_explanation": None,
                "protein1_uniprot_id": p1,
                "protein2_uniprot_id": p2,
                "protein1_seq": pair.protein1_seq or "",
                "protein2_seq": pair.protein2_seq or ""
            })

    return results
