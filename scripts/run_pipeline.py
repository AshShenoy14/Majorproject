import sys
import os
import torch
import pandas as pd
import argparse
import gc
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, MODELS_DIR
from src.data.sequence_manager import SequenceManager
from src.data.feature_extraction import ESMFeatureExtractor
from src.training.train_sequence_model import train as train_seq
from src.training.train_graph_model import train as train_graph
from src.training.train_ensemble import train_ensemble
from torch_geometric.data import Data

def run_pipeline(limit_data: int = None):
    print("=== Starting Real Data Pipeline ===")
    
    # 1. Load Data
    print("Loading train.csv...")
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        print("Error: train.csv not found.")
        return
        
    df = pd.read_csv(train_path)
    if limit_data:
        print(f"Limiting to first {limit_data} interactions for quick setup...")
        df = df.head(limit_data)
        
    # Get Unique Proteins
    proteins = set(df["protein1"].unique()) | set(df["protein2"].unique())
    print(f"Found {len(proteins)} unique proteins.")
    
    # 2. Fetch Sequences
    print("--- Fetching Sequences ---")
    seq_manager = SequenceManager()
    sequences = seq_manager.get_sequences(list(proteins))
    
    missing = len(proteins) - len(sequences)
    if missing > 0:
        print(f"Warning: {missing} proteins missing sequences.")
        
    # 3. Extract Embeddings
    # 3. Extract Embeddings
    print("--- Extracting Embeddings ---")
    # Check if embeddings already exist?
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    requires_extraction = True
    embeddings = {}

    if emb_path.exists() and not limit_data:
        print("Embeddings file exists. Checking coverage...")
        try:
            loaded_embeddings = torch.load(emb_path, weights_only=False)
            # Keep embeddings in float16 to save RAM — conversion done per-sample during training
            loaded_keys = set(loaded_embeddings.keys())
            missing_count = len(proteins - loaded_keys)
            
            if missing_count > len(proteins) * 0.1: # If more than 10% missing
                print(f"Embedding coverage low. Missing {missing_count} proteins. Regenerating...")
                # Delete old file before regenerating
                os.remove(emb_path)
                requires_extraction = True
            else:
                 print(f"Embedding coverage sufficient. Skipping extraction.")
                 embeddings = loaded_embeddings
                 requires_extraction = False
        except Exception as e:
            print(f"Error loading embeddings: {e}. Regenerating...")
            # Delete corrupted file
            if emb_path.exists():
                os.remove(emb_path)
            requires_extraction = True
            
    if requires_extraction:
        # Filter sequences to only those we have
        valid_sequences = {k: v for k, v in sequences.items() if v}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Increased batch size for speed
        extractor = ESMFeatureExtractor(device=device)
        embeddings = extractor.get_embeddings(valid_sequences, batch_size=8)
        
        if not limit_data:
            print("Saving embeddings to file...")
            torch.save(embeddings, emb_path)
            print("Embeddings saved.")
    
    # 4. Build Graph
    print("--- Building PPI Graph ---")
    # Map proteins to indices
    valid_proteins = sorted(list(embeddings.keys()))
    node_mapping = {p: i for i, p in enumerate(valid_proteins)}
    
    # Edges
    src = []
    dst = []
    for _, row in df.iterrows():
        if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
            # Undirected graph usually? Or strictly from dataset
            # PPI is usually undirected.
            u, v = node_mapping[row["protein1"]], node_mapping[row["protein2"]]
            src.append(u)
            dst.append(v)
            # Add reverse for undirected message passing
            src.append(v)
            dst.append(u)
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Node Features (float32 for model compatibility)
    # ESM-2 t6_8M_UR50D has embedding dimension 320
    embedding_dim = 320
    x = torch.empty(len(valid_proteins), embedding_dim, dtype=torch.float32)

    for i, p in enumerate(valid_proteins):
        emb = embeddings[p].float()
        # If per-residue embeddings are provided [seq_len, 320], mean pool them
        if emb.dim() > 1:
            emb = emb.mean(dim=0)
        
        # Safety check: if for some reason the dimension is wrong (e.g. if emb was already 1D but not 320)
        if emb.shape[0] != embedding_dim:
             # This should only happen if sequences are very short or something is corrupted
             # but we'll pad/truncate just in case to prevent crash
             new_emb = torch.zeros(embedding_dim)
             copy_size = min(emb.shape[0], embedding_dim)
             new_emb[:copy_size] = emb[:copy_size]
             emb = new_emb
             
        x[i] = emb
    
    data = Data(x=x, edge_index=edge_index)
    
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    
    # Save temp files before deleting embeddings (for limit_data case)
    if limit_data:
        temp_emb_path = PROCESSED_DATA_DIR / "temp_embeddings.pt"
        torch.save(embeddings, temp_emb_path)
        temp_graph_path = PROCESSED_DATA_DIR / "temp_graph.pt"
        torch.save(data, temp_graph_path)
        torch.save(node_mapping, str(temp_graph_path).replace(".pt", "_mapping.pt"))
    else:
        torch.save(data, graph_path)
        torch.save(node_mapping, mapping_path)
    
    # Always clean up embeddings after graph is built and saved to free memory
    del embeddings
    gc.collect()
    print("--- Training Models ---")
    
    # Paths
    seq_model_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    
    # Train Sequence Model
    if limit_data:
        # Don't overwrite main models if running limited test
        seq_model_path = PROJECT_ROOT / "models" / "sequence_model_test.pth"
        graph_model_path = PROJECT_ROOT / "models" / "graph_model_test.pth"
        
    print("Training Sequence Model...")
    # We need to ensure we pass the embeddings we just generated
    # The training script loads embeddings from file usually, but we can Modify it or 
    # Just save it temporarily if limit_data
    if limit_data:
        temp_emb_path = PROCESSED_DATA_DIR / "temp_embeddings.pt"
        train_seq(epochs=2, embedding_path=str(temp_emb_path))
    else:
        train_seq(epochs=50, embedding_path=str(emb_path))
        
    # Train Graph Model
    print("Training Graph Model...")
    if limit_data:
        temp_graph_path = PROCESSED_DATA_DIR / "temp_graph.pt"
        # Graph training script expects mapping file next to graph
        train_graph(epochs=10, graph_path=str(temp_graph_path))
    else:
        train_graph(epochs=100, graph_path=str(graph_path))
        
        # --- Phase 4: Ensemble Meta-Learner ---
        print("\n=== Phase 4: Training Ensemble Meta-Learner (XGBoost) ===")
        seq_model_path = MODELS_DIR / "sequence_model_best.pth"
        graph_model_path = MODELS_DIR / "graph_model_best.pth"
        train_ensemble(str(seq_model_path), str(graph_model_path), str(graph_path))
        
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of interactions for testing")
    args = parser.parse_args()
    
    run_pipeline(args.limit)
