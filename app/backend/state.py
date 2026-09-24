"""Global model/manager state and startup loading for the TransGraph-PPI API.

Routers import this module (not its names) so that reassignments made during
load_system() (e.g. `explainer`) are visible everywhere: use `state.explainer`,
`state.models`, etc. rather than `from app.backend.state import explainer`.
"""
import sys

import torch
import torch.nn.functional as F
import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.models.sequence_model import SequencePPIModel
from src.utils.esm_config import ESM_EMBED_DIM
from src.models.ensemble_model import PPIEnsemble
from src.data.feature_extraction import ESMFeatureExtractor
from src.data.sequence_manager import SequenceManager
from src.data.target_manager import TargetManager
from src.data.id_mapper import IDMapper
from src.analysis.explainability import PPIExplainer
from src.analysis.network_analysis import NetworkAnalyzer
from src.analysis.mutation_analyzer import MutationAnalyzer
from src.analysis.biological_managers import BiologicalManager
from src.analysis.protein_assistant import ProteinAssistant
from src.utils.bio_encoder import BioFeatureEncoder

# Global State
models = {}
managers = {}
data_cache = {}
analyzers = {}
explainer = None  # Global explainer instance


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


async def load_system():
    global explainer
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
        if not seq_path.exists():
            raise RuntimeError(f"CRITICAL SAFETY ERROR: Sequence model checkpoint missing at {seq_path}. Fallback to random weights is strictly prohibited.")
        try:
            models["seq_model"] = SequencePPIModel(input_dim=ESM_EMBED_DIM).to(device)
            models["seq_model"].load_state_dict(torch.load(seq_path, map_location=device))
            models["seq_model"].eval()
            print("Sequence Model loaded successfully from checkpoint.")
        except Exception as e:
            raise RuntimeError(f"CRITICAL SAFETY ERROR: Failed to load sequence model from {seq_path}: {e}")

        # Graph Model
        from src.models.graph_model import SAGELinkPredictor, GINLinkPredictor
        graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
        graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
        if not graph_path.exists() or not graph_data_path.exists():
            raise RuntimeError(f"CRITICAL SAFETY ERROR: Graph model checkpoint ({graph_path}) or data ({graph_data_path}) missing. Fallbacks strictly prohibited.")
        try:
            data_cache["graph"] = torch.load(graph_data_path, weights_only=False).to(device)
            in_channels = data_cache["graph"].x.shape[1]
            state_dict = torch.load(graph_path, map_location=device)
            is_gin = any("convs" in k for k in state_dict.keys())

            if is_gin:
                print("Detected GIN architecture for Graph Model.")
                models["graph_model"] = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
            else:
                print("Detected SAGEConv architecture for Graph Model.")
                models["graph_model"] = SAGELinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)

            models["graph_model"].load_state_dict(state_dict)
            models["graph_model"].eval()
            print("Graph Model loaded successfully from checkpoint.")
        except Exception as e:
            raise RuntimeError(f"CRITICAL SAFETY ERROR: Failed to load graph model from {graph_path}: {e}")

        map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
        if map_path.exists():
            data_cache["mapping"] = torch.load(map_path, weights_only=False)
            # PageRank is the last of the 3 topological node features appended to the ESM embeddings in
            # src/data/graph_construction.py ([degree_centrality, clustering, pagerank]); expose it as computed there.
            pr_col = data_cache["graph"].x[:, -1].cpu().tolist()
            data_cache["pagerank"] = {pid: pr_col[idx] for pid, idx in data_cache["mapping"].items()}

        # 3. Load Ensemble model
        ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
        if not ensemble_path.exists():
            raise RuntimeError(f"CRITICAL SAFETY ERROR: Ensemble meta-learner missing at {ensemble_path}. Fallbacks strictly prohibited.")
        try:
            models["ensemble"] = PPIEnsemble(str(ensemble_path))
            print("Loaded Hybrid Ensemble meta-learner.")

            # Initialize Explainer
            print("Initializing SHAP explainer...")
            explainer = PPIExplainer(str(ensemble_path))
        except Exception as e:
            raise RuntimeError(f"CRITICAL SAFETY ERROR: Failed to load ensemble model from {ensemble_path}: {e}")

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

            analyzers["mutation"] = MutationAnalyzer(models["seq_model"], models["esm"], managers["bio"], managers["bio_encoder"])
            analyzers["hotspot"] = HotspotAnalyzer(models["seq_model"], models["esm"], managers["bio"], managers["bio_encoder"])
            analyzers["residue_graph"] = ResidueGraphGenerator(device=device)
            print("Mutation and Novel Analyzers Ready.")

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
