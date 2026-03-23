"""
Compare models with leakage detection and honest metrics.
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    roc_curve
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.analysis.explainability import PPIExplainer


def check_leakage():
    """Verify no protein leakage between train and test."""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    train_proteins = (set(train_df['protein1']) |
                      set(train_df['protein2']))
    test_proteins = (set(test_df['protein1']) |
                     set(test_df['protein2']))

    overlap = train_proteins & test_proteins
    overlap_ratio = len(overlap) / len(test_proteins) if test_proteins else 0

    print(f"\n=== Leakage Check ===")
    print(f"Train proteins: {len(train_proteins)}")
    print(f"Test proteins:  {len(test_proteins)}")
    print(f"Overlap:        {len(overlap)} ({overlap_ratio:.1%})")

    if overlap_ratio > 0.5:
        print("⚠️  HIGH OVERLAP — metrics are INFLATED!")
        print("   Run: python src/data/split_dataset.py")
        print("   Then retrain both models.\n")
        return False
    elif overlap_ratio > 0:
        print(f"⚠️  Some overlap exists ({len(overlap)} proteins)")
        return True
    else:
        print("✓  CLEAN — no protein leakage")
        return True


def check_graph_leakage():
    """Verify test edges are NOT in the message-passing graph."""
    graph_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not all(p.exists() for p in [graph_path, map_path, test_path]):
        print("Cannot check graph leakage — files missing")
        return

    graph_data = torch.load(graph_path, weights_only=False)
    node_mapping = torch.load(map_path, weights_only=False)
    test_df = pd.read_csv(test_path)

    # Get graph edges as set of pairs
    ei = graph_data.edge_index.numpy()
    graph_edges = set()
    for i in range(ei.shape[1]):
        pair = (min(ei[0, i], ei[1, i]), max(ei[0, i], ei[1, i]))
        graph_edges.add(pair)

    # Check how many test positive edges are in graph
    leaked = 0
    total_test_pos = 0

    for _, row in test_df[test_df['label'] == 1].iterrows():
        p1, p2 = row['protein1'], row['protein2']
        if p1 in node_mapping and p2 in node_mapping:
            u, v = node_mapping[p1], node_mapping[p2]
            pair = (min(u, v), max(u, v))
            total_test_pos += 1
            if pair in graph_edges:
                leaked += 1

    print(f"\n=== Graph Edge Leakage ===")
    print(f"Test positive edges: {total_test_pos}")
    print(f"Found in graph:      {leaked}")

    if leaked > 0:
        print(f"⚠️  {leaked} test edges LEAKED into graph!")
        print("   Retrain with: python src/training/train_graph.py")
    else:
        print("✓  CLEAN — no test edges in graph")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...\n")

    # Check for leakage FIRST
    is_clean = check_leakage()
    check_graph_leakage()

    if not is_clean:
        print("\n" + "=" * 60)
        print("FIX LEAKAGE BEFORE TRUSTING THESE METRICS")
        print("=" * 60 + "\n")

    # Load test data
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    embeddings = torch.load(PROCESSED_DATA_DIR / "embeddings.pt",
                            weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v
                  for k, v in embeddings.items()}
    node_mapping = torch.load(PROCESSED_DATA_DIR / "ppi_graph_mapping.pt",
                              weights_only=False)

    # Auto-detect embedding dimension from loaded embeddings
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1] if sample_emb.dim() > 1 else sample_emb.shape[0]
    print(f"Detected embedding dimension: {input_dim}")

    # Load sequence model
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    seq_model.load_state_dict(torch.load(
        PROJECT_ROOT / "models" / "sequence_model_best.pth",
        map_location=device
    ))
    seq_model.eval()

    graph_data = torch.load(
        PROCESSED_DATA_DIR / "ppi_graph.pt", weights_only=False
    ).to(device)

    # Load graph model
    config = torch.load(
        PROJECT_ROOT / "models" / "graph_model_config.pt",
        map_location="cpu", weights_only=False
    )
    graph_model = GATLinkPredictor(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        heads=config.get("heads", 4),
    ).to(device)
    graph_model.load_state_dict(torch.load(
        PROJECT_ROOT / "models" / "graph_model_best.pth",
        map_location=device
    ))
    graph_model.eval()

    # Load ensemble
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    ensemble_model = None
    if ensemble_path.exists():
        import joblib
        ensemble_model = joblib.load(ensemble_path)

    # Separate: proteins IN graph vs NOT in graph
    known_rows = []
    novel_rows = []

    for _, row in test_df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        if p1 not in embeddings or p2 not in embeddings:
            continue

        in_graph = (p1 in node_mapping and p2 in node_mapping)
        if in_graph:
            known_rows.append(row)
        else:
            novel_rows.append(row)

    print(f"\nTest split: {len(known_rows)} known-protein pairs, "
          f"{len(novel_rows)} novel-protein pairs")

    # Pre-compute GAT node embeddings
    with torch.no_grad():
        z_gat = graph_model.encode(
            graph_data.x, graph_data.edge_index
        )
        # Using batch size for sequence embeddings mapped to gat space
        all_prots = list(embeddings.keys())
        prot_to_idx = {p: i for i, p in enumerate(all_prots)}
        emb_matrix = []
        for p in all_prots:
            e = embeddings[p]
            emb_matrix.append(e.mean(0) if e.dim() > 1 else e)
        emb_matrix = torch.stack(emb_matrix)

        fallback_embs_list = []
        bs = 1000
        for i in range(0, len(emb_matrix), bs):
            fblk = graph_model.encode_sequences(emb_matrix[i:i+bs].to(device))
            fallback_embs_list.append(fblk)
        fallback_embs = torch.cat(fallback_embs_list, dim=0)

    # Evaluate on BOTH subsets
    for subset_name, rows in [("Known Proteins", known_rows),
                              ("Novel Proteins", novel_rows),
                              ("All Test", known_rows + novel_rows)]:
        if not rows:
            print(f"\n{subset_name}: No samples")
            continue

        df_subset = pd.DataFrame(rows)
        labels = df_subset['label'].values

        # Sequence predictions
        seq_preds = []
        graph_preds = []

        with torch.no_grad():
            bs = 128
            for i in tqdm(range(0, len(df_subset), bs), desc=f"Predicting {subset_name}"):
                batch_rows = df_subset.iloc[i:i+bs]
                
                # Fetch Esm embeddings
                e1_batch = []
                e2_batch = []
                b1_idx = []
                b2_idx = []
                b1_known = []
                b2_known = []
                
                for _, row in batch_rows.iterrows():
                    p1, p2 = row['protein1'], row['protein2']
                    e1 = embeddings[p1]
                    e2 = embeddings[p2]
                    e1_batch.append(e1.mean(0) if e1.dim() > 1 else e1)
                    e2_batch.append(e2.mean(0) if e2.dim() > 1 else e2)
                    
                    if p1 in node_mapping:
                        b1_known.append(True)
                        b1_idx.append(node_mapping[p1])
                    else:
                        b1_known.append(False)
                        b1_idx.append(prot_to_idx[p1])
                        
                    if p2 in node_mapping:
                        b2_known.append(True)
                        b2_idx.append(node_mapping[p2])
                    else:
                        b2_known.append(False)
                        b2_idx.append(prot_to_idx[p2])

                e1_tens = torch.stack(e1_batch).to(device)
                e2_tens = torch.stack(e2_batch).to(device)
                
                # Seq Pred
                out = seq_model(e1_tens, e2_tens)
                seq_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
                
                # Graph Pred
                z1_batch = []
                z2_batch = []
                for j in range(len(b1_idx)):
                    if b1_known[j]:
                        z1_batch.append(z_gat[b1_idx[j]])
                    else:
                        z1_batch.append(fallback_embs[b1_idx[j]])
                        
                    if b2_known[j]:
                        z2_batch.append(z_gat[b2_idx[j]])
                    else:
                        z2_batch.append(fallback_embs[b2_idx[j]])
                
                z1_tens = torch.stack(z1_batch).to(device)
                z2_tens = torch.stack(z2_batch).to(device)
                
                g_out = graph_model.decode_from_embeddings(z1_tens, z2_tens)
                graph_preds.extend(torch.sigmoid(g_out).cpu().numpy().flatten())

        seq_preds = np.array(seq_preds)
        graph_preds = np.array(graph_preds)
        
        # Predict Ensemble (With 5 features including disagreement)
        ens_preds = None
        if ensemble_model:
            conf_seq = np.abs(seq_preds - 0.5)
            conf_gat = np.abs(graph_preds - 0.5)
            disagreement = np.abs(seq_preds - graph_preds)
            X = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat, disagreement))
            try:
                ens_preds = ensemble_model.predict_proba(X)[:, 1]
            except Exception:
                # Fallback to 4 features just in case
                X_fallback = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat))
                ens_preds = ensemble_model.predict_proba(X_fallback)[:, 1]

        # Metrics
        print(f"\n=== {subset_name} (n={len(labels)}) ===")
        results = []
        
        def calc_met(ypreds):
            binary = (ypreds > 0.5).astype(int)
            return [
                accuracy_score(labels, binary),
                precision_score(labels, binary, zero_division=0),
                recall_score(labels, binary, zero_division=0),
                f1_score(labels, binary, zero_division=0),
                roc_auc_score(labels, ypreds),
                average_precision_score(labels, ypreds)
            ]
            
        results.append(["Sequence"] + calc_met(seq_preds))
        results.append(["GAT"] + calc_met(graph_preds))
        if ens_preds is not None:
            results.append(["Ensemble"] + calc_met(ens_preds))

        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))


if __name__ == "__main__":
    evaluate()