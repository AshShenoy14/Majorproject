import sys
import os
import torch
import pandas as pd
import argparse
import gc
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, MODELS_DIR, CHECKPOINT_DIR
from src.data.sequence_manager import SequenceManager
from src.data.feature_extraction import ESMFeatureExtractor
from src.training.train_sequence_model import train as train_seq
from src.training.train_graph_model import train as train_graph
from src.training.train_ensemble import train_ensemble
from torch_geometric.data import Data


def run_pipeline(limit_data: int = None):
    print("=" * 60)
    print("  PPI Prediction Pipeline (P Test)")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # ================================================================
    # 1. Load Data
    # ================================================================
    print("\n[Step 1/6] Loading training data...")
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists():
        print("Error: train.csv not found at", train_path)
        return

    df = pd.read_csv(train_path)

    # Also load val/test to ensure ALL proteins get embeddings
    all_dfs = [df]
    for p in [val_path, test_path]:
        if p.exists():
            all_dfs.append(pd.read_csv(p))
            print(f"  Loaded {p.name}: {len(all_dfs[-1])} rows")

    all_data = pd.concat(all_dfs, ignore_index=True)

    if limit_data:
        print(f"  Limiting train to first {limit_data} interactions for quick test...")
        df = df.head(limit_data)
        # Still use all proteins for embeddings from limited set
        all_data = df

    # Get ALL unique proteins across train/val/test
    proteins = set(all_data["protein1"].unique()) | set(all_data["protein2"].unique())
    print(f"  Found {len(proteins)} unique proteins across all splits.")
    print(f"  Train: {len(df)} interactions")

    # ================================================================
    # 2. Fetch Sequences
    # ================================================================
    print("\n[Step 2/6] Fetching protein sequences...")
    seq_manager = SequenceManager()
    sequences = seq_manager.get_sequences(list(proteins))

    valid_sequences = {k: v for k, v in sequences.items() if v}
    missing = len(proteins) - len(valid_sequences)
    print(f"  Retrieved: {len(valid_sequences)} sequences")
    if missing > 0:
        print(f"  Warning: {missing} proteins missing sequences.")

    # ================================================================
    # 3. Extract Embeddings
    # ================================================================
    print("\n[Step 3/6] Extracting ESM embeddings...")
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    requires_extraction = True
    embeddings = {}

    if emb_path.exists() and not limit_data:
        print("  Embeddings file exists. Checking coverage...")
        try:
            loaded_embeddings = torch.load(emb_path, weights_only=False)
            loaded_keys = set(loaded_embeddings.keys())
            missing_proteins = proteins - loaded_keys
            missing_count = len(missing_proteins)

            if missing_count > len(proteins) * 0.1:
                print(f"  Coverage low — missing {missing_count}/{len(proteins)} proteins. Regenerating...")
                os.remove(emb_path)
                requires_extraction = True
            elif missing_count > 0:
                # Extract only missing proteins, merge with existing
                print(f"  Extracting {missing_count} missing protein embeddings...")
                missing_seqs = {k: v for k, v in valid_sequences.items() if k in missing_proteins and v}
                if missing_seqs:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    extractor = ESMFeatureExtractor(device=device)
                    new_embeddings = extractor.get_embeddings(missing_seqs, batch_size=8)
                    loaded_embeddings.update(new_embeddings)
                    torch.save(loaded_embeddings, emb_path)
                    print(f"  Merged {len(new_embeddings)} new embeddings. Total: {len(loaded_embeddings)}")
                    del new_embeddings
                embeddings = loaded_embeddings
                requires_extraction = False
            else:
                print(f"  Full coverage ({len(loaded_keys)} proteins). Skipping extraction.")
                embeddings = loaded_embeddings
                requires_extraction = False
        except Exception as e:
            print(f"  Error loading embeddings: {e}. Regenerating...")
            if emb_path.exists():
                os.remove(emb_path)
            requires_extraction = True

    if requires_extraction:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = ESMFeatureExtractor(device=device)
        embeddings = extractor.get_embeddings(valid_sequences, batch_size=8)
        print(f"  Extracted {len(embeddings)} embeddings.")

        if not limit_data:
            print("  Saving embeddings...")
            torch.save(embeddings, emb_path)
            print(f"  Saved to {emb_path}")

    # ================================================================
    # 4. Build PPI Graph
    # ================================================================
    print("\n[Step 4/6] Building PPI graph...")

    # Mean-pool per-residue embeddings to fixed-size vectors for graph node features
    pooled_embeddings = {}
    for k, v in embeddings.items():
        if v.dim() > 1:
            pooled_embeddings[k] = v.mean(dim=0).float()
        else:
            pooled_embeddings[k] = v.float()

    valid_proteins = sorted(list(pooled_embeddings.keys()))
    node_mapping = {p: i for i, p in enumerate(valid_proteins)}

    # Build edges from ALL splits (train + val + test) for message passing
    # Only positive interactions contribute to graph structure
    src = []
    dst = []
    edge_pairs_seen = set()

    for split_df in all_dfs:
        for _, row in split_df.iterrows():
            p1, p2 = row["protein1"], row["protein2"]
            if p1 in node_mapping and p2 in node_mapping:
                u, v = node_mapping[p1], node_mapping[p2]
                # Only add positive edges to graph structure
                if row.get("label", 1) == 1:
                    pair = (min(u, v), max(u, v))
                    if pair not in edge_pairs_seen:
                        edge_pairs_seen.add(pair)
                        src.extend([u, v])
                        dst.extend([v, u])

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node features
    embedding_dim = next(iter(pooled_embeddings.values())).shape[0]
    x = torch.zeros(len(valid_proteins), embedding_dim, dtype=torch.float32)
    for i, p in enumerate(valid_proteins):
        x[i] = pooled_embeddings[p]

    data = Data(x=x, edge_index=edge_index)

    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges} (undirected pairs: {len(edge_pairs_seen)})")
    print(f"  Node feature dim: {embedding_dim}")

    # Save graph and mapping
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    mapping_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"

    if limit_data:
        graph_path = PROCESSED_DATA_DIR / "temp_graph.pt"
        mapping_path = PROCESSED_DATA_DIR / "temp_graph_mapping.pt"
        # Save temp embeddings for limited training
        temp_emb_path = PROCESSED_DATA_DIR / "temp_embeddings.pt"
        torch.save(embeddings, temp_emb_path)

    torch.save(data, graph_path)
    torch.save(node_mapping, mapping_path)
    print(f"  Graph saved to {graph_path}")
    print(f"  Mapping saved to {mapping_path}")

    # Free memory
    del pooled_embeddings, edge_pairs_seen
    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================
    # 5. Train Models
    # ================================================================
    print("\n[Step 5/6] Training models...")

    seq_model_path = MODELS_DIR / "sequence_model_best.pth"
    graph_model_path = MODELS_DIR / "graph_model_best.pth"

    # --- 5a. Train Sequence Model ---
    print("\n--- Training Sequence Model ---")
    if limit_data:
        temp_emb_path = PROCESSED_DATA_DIR / "temp_embeddings.pt"
        train_seq(epochs=2, embedding_path=str(temp_emb_path))
    else:
        emb_path_str = str(PROCESSED_DATA_DIR / "embeddings.pt")
        train_seq(epochs=30, embedding_path=emb_path_str)

    # Verify sequence model was saved
    if seq_model_path.exists():
        print(f"  Sequence model saved: {seq_model_path}")
    else:
        print("  WARNING: Sequence model not found after training!")

    # --- 5b. Train Graph Model ---
    print("\n--- Training Graph Model ---")
    if limit_data:
        temp_graph_path = PROCESSED_DATA_DIR / "temp_graph.pt"
        train_graph(epochs=10, graph_path=str(temp_graph_path))
    else:
        train_graph(epochs=100, graph_path=str(graph_path))

    # Verify graph model was saved
    if graph_model_path.exists():
        print(f"  Graph model saved: {graph_model_path}")
    else:
        print("  WARNING: Graph model not found after training!")

    # ================================================================
    # 6. Train Ensemble
    # ================================================================
    print("\n[Step 6/6] Training ensemble...")

    if limit_data:
        print("  Skipping ensemble in limited mode (needs full val data).")
    else:
        if seq_model_path.exists() and graph_model_path.exists():
            train_ensemble(
                str(seq_model_path),
                str(graph_model_path),
                str(graph_path)
            )
            ensemble_path = MODELS_DIR / "ensemble_model.pkl"
            if ensemble_path.exists():
                print(f"  Ensemble model saved: {ensemble_path}")
            else:
                print("  WARNING: Ensemble model not found after training!")
        else:
            print("  ERROR: Cannot train ensemble — base models missing.")
            if not seq_model_path.exists():
                print(f"    Missing: {seq_model_path}")
            if not graph_model_path.exists():
                print(f"    Missing: {graph_model_path}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  Pipeline Complete (P Test)")
    print("=" * 60)
    print("\n  Model Status:")

    models_to_check = [
        ("Sequence Model", MODELS_DIR / "sequence_model_best.pth"),
        ("Graph Model", MODELS_DIR / "graph_model_best.pth"),
        ("Ensemble Model", MODELS_DIR / "ensemble_model.pkl"),
    ]

    for name, path in models_to_check:
        if path.exists():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    ✓ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"    ✗ {name}: NOT FOUND")

    print("\n  Next steps:")
    print("    python src/analysis/compare_models.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of interactions for testing")
    args = parser.parse_args()

    run_pipeline(args.limit)