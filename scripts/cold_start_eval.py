"""
Simulated cold-start / semi-inductive evaluation.

The production /predict cold-start path (insert_novel_node_knn in app/backend/main.py) is never
actually exercised in the standard test-set evaluation, because every protein in train/val/test
already has a node in ppi_graph.pt (only edges are restricted to train positives -- see
"Graph" in README's Dataset & Evaluation Protocol section). So the code that is supposed to
handle a genuinely unseen protein has shipped without ever being scored against ground truth.

This script builds that missing evaluation:
  1. Physically removes a random sample of proteins ("novel" proteins) from the trained graph
     (drops their nodes and every edge touching them; remaining nodes are reindexed).
  2. Re-runs GraphSAGE.encode() on this pruned graph using the ALREADY-TRAINED weights (no
     retraining anywhere in this script).
  3. For each test.csv pair where exactly one endpoint is a "novel" protein and the other
     endpoint survives in the pruned graph, reproduces the exact production cold-start code path:
     insert_novel_node_knn() finds the novel protein's k nearest surviving nodes by ESM cosine
     similarity, GraphSAGE decodes edges to each neighbor, and the mean is taken as graph_prob.
  4. Combines with the sequence model and the existing (unretrained) 7-feature XGBoost ensemble,
     exactly as compare_models.py / main.py do.
  5. Reports metrics on this semi-cold-start subset, alongside the SAME pairs metrics under
     the normal (warm, transductive) path for a direct before/after comparison.

Caveats (read before quoting these numbers as inductive generalization):
  - This is not a from-scratch protein-disjoint retrain. The sequence model, GraphSAGE weights and
    the XGBoost meta-learner were all originally fit with these novel proteins data available
    (their embeddings were seen by the sequence model training data if they appeared in train.csv
    pairs, and their edges were part of the graph GraphSAGE trained its weights on). Removing a node
    post-hoc changes message passing for surviving neighbors but does not undo what the weights
    already learned from that protein historical connectivity.
  - Surviving nodes topological features (degree/clustering/PageRank, baked into ppi_graph.pt at
    construction time) are not recomputed after pruning, so they are mildly stale.
  - This should be reported as a simulated, post-hoc test of the cold-start inference code path
    described in README/validation_record as exploratory and unevaluated -- a real step up from
    never tested, but weaker evidence than a genuine held-out-protein retrain.
"""
import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from src.utils.paths import PROCESSED_DATA_DIR as P, MODELS_DIR as M, PROJECT_ROOT
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor
from src.models.ensemble_model import PPIEnsemble


def insert_novel_node_knn(novel_emb, existing_embs, k=2):
    """Reference (unvectorized) version, kept identical to app/backend/main.py's function so a unit
    test can confirm it agrees with the batched version below on a small case."""
    if not existing_embs:
        return []
    novel_emb_cpu = novel_emb.cpu().float()
    sims = {nid: F.cosine_similarity(novel_emb_cpu.unsqueeze(0), e.cpu().float().unsqueeze(0)).item()
            for nid, e in existing_embs.items()}
    return [nid for nid, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:k]]


class BatchKNN:
    """Vectorized re-implementation of insert_novel_node_knn: same top-k-by-cosine-similarity result,
    computed with one matrix multiply against a stacked matrix of existing embeddings instead of a
    per-candidate Python-level call. Only used to make this evaluation script tractable at scale;
    app/backend/main.py's own per-request function is untouched."""

    def __init__(self, existing_embs: dict):
        self.ids = list(existing_embs.keys())
        mat = torch.stack([existing_embs[i].float() for i in self.ids])
        self.mat_norm = F.normalize(mat, dim=1)

    def query(self, novel_emb, k=2):
        q = F.normalize(novel_emb.float().unsqueeze(0), dim=1)
        sims = (self.mat_norm @ q.T).squeeze(1)
        top = torch.topk(sims, k=min(k, len(self.ids))).indices.tolist()
        return [self.ids[i] for i in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_novel", type=int, default=400, help="number of proteins to remove from the graph")
    ap.add_argument("--k", type=int, default=2, help="KNN neighbors for cold-start (matches main.py default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "assets" / "evaluation" / "cold_start_eval.json"))
    args = ap.parse_args()

    dev = torch.device("cpu")
    rng = np.random.default_rng(args.seed)

    emb = {k_: (v.float() if v.dtype == torch.float16 else v) for k_, v in torch.load(P / "embeddings.pt", weights_only=False).items()}
    mapping = torch.load(P / "ppi_graph_mapping.pt", weights_only=False)
    graph = torch.load(P / "ppi_graph.pt", weights_only=False)
    esm_dim = graph.x.shape[1] - 3
    test_df = pd.read_csv(P / "test.csv")

    sm = SequencePPIModel(input_dim=esm_dim, hidden_dim=1024)
    sm.load_state_dict(torch.load(M / "sequence_model_best.pth", map_location=dev))
    sm.eval()
    gm = SAGELinkPredictor(in_channels=graph.x.shape[1], hidden_channels=256)
    gm.load_state_dict(torch.load(M / "graph_model_best.pth", map_location=dev, weights_only=False))
    gm.eval()
    ens = PPIEnsemble(str(M / "ensemble_model.pkl"))  # loads meta_model + graph_calibrator, matching main.py

    inv_mapping = {v: k_ for k_, v in mapping.items()}
    all_proteins = list(mapping.keys())
    novel_proteins = set(rng.choice(all_proteins, size=min(args.n_novel, len(all_proteins)), replace=False).tolist())
    print(f"Removing {len(novel_proteins)} proteins from the trained graph as novel (unseen-at-inference) proteins.")

    # prune the graph: drop novel-protein nodes + any edge touching them, reindex survivors
    keep_idx = [idx for pid, idx in mapping.items() if pid not in novel_proteins]
    keep_idx_set = set(keep_idx)
    old_to_new = {old: new for new, old in enumerate(sorted(keep_idx))}
    x_pruned = graph.x[sorted(keep_idx)]
    ei = graph.edge_index.numpy()
    mask = np.isin(ei[0], list(keep_idx_set)) & np.isin(ei[1], list(keep_idx_set))
    ei_pruned = ei[:, mask]
    ei_pruned = np.vectorize(old_to_new.get)(ei_pruned)
    edge_index_pruned = torch.tensor(ei_pruned, dtype=torch.long)
    new_mapping = {inv_mapping[old]: new for old, new in old_to_new.items()}
    print(f"Pruned graph: {x_pruned.shape[0]} nodes (was {graph.x.shape[0]}), {edge_index_pruned.shape[1]} directed edges (was {graph.edge_index.shape[1]}).")

    existing_embeddings = {pid: x_pruned[idx, :esm_dim] for pid, idx in new_mapping.items()}
    knn = BatchKNN(existing_embeddings)

    # sanity check: vectorized KNN agrees with the reference (production) implementation
    sample_ids = list(existing_embeddings.keys())[:5]
    probe_pid = sample_ids[0]
    ref = insert_novel_node_knn(emb[probe_pid], {i: existing_embeddings[i] for i in sample_ids[1:]}, k=2)
    fast = BatchKNN({i: existing_embeddings[i] for i in sample_ids[1:]}).query(emb[probe_pid], k=2)
    assert set(ref) == set(fast), f"vectorized KNN disagrees with reference: {ref} vs {fast}"
    print("Vectorized KNN verified against reference implementation on a probe case.")

    with torch.no_grad():
        z_pruned = gm.encode(x_pruned, edge_index_pruned)
        z_full = gm.encode(graph.x, graph.edge_index)

    # find eligible test pairs: exactly one endpoint novel, the other a survivor
    def status(pid):
        if pid in novel_proteins:
            return "novel"
        return "survivor" if pid in new_mapping else "absent"

    elig = []
    for _, row in test_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        s1, s2 = status(p1), status(p2)
        if s1 == "absent" or s2 == "absent":
            continue
        if (s1 == "novel") != (s2 == "novel"):
            novel_p, known_p = (p1, p2) if s1 == "novel" else (p2, p1)
            elig.append((novel_p, known_p, int(label)))
    print(f"Eligible semi-cold-start test pairs (exactly one endpoint held out): {len(elig)}")

    elig = [e for e in elig if e[0] in emb and e[1] in emb]
    print(f"Eligible pairs with embeddings present: {len(elig)}", flush=True)

    # batch the sequence model over all eligible pairs at once
    e1_batch = torch.stack([emb[p[0]] for p in elig]).float()
    e2_batch = torch.stack([emb[p[1]] for p in elig]).float()
    with torch.no_grad():
        seq_probs = torch.sigmoid(sm(e1_batch, e2_batch)).squeeze(-1).numpy()
    print("Sequence model scored.", flush=True)

    cold_graph_probs, warm_graph_probs, labels = [], [], []
    for i, (novel_p, known_p, label) in enumerate(elig):
        if i % 200 == 0:
            print(f"  graph scoring {i}/{len(elig)}", flush=True)
        nb = knn.query(emb[novel_p], k=args.k)
        if not nb:
            cold_graph_probs.append(np.nan); warm_graph_probs.append(np.nan); labels.append(label)
            continue
        known_idx = new_mapping[known_p]
        nb_idx = [new_mapping[n] for n in nb]
        src = torch.tensor(nb_idx, dtype=torch.long)
        dst = torch.tensor([known_idx] * len(nb_idx), dtype=torch.long)
        with torch.no_grad():
            g_out = gm.decode(z_pruned, src, dst)
            cold_p = torch.sigmoid(g_out).mean().item()
        i1, i2 = mapping[novel_p], mapping[known_p]
        with torch.no_grad():
            warm_out = gm.decode(z_full, torch.tensor([i1]), torch.tensor([i2]))
            warm_p = torch.sigmoid(warm_out).item()
        cold_graph_probs.append(cold_p); warm_graph_probs.append(warm_p); labels.append(label)

    valid = ~np.isnan(cold_graph_probs)
    seq_probs = seq_probs[valid]
    cold_graph_probs = np.array(cold_graph_probs)[valid]
    warm_graph_probs = np.array(warm_graph_probs)[valid]
    labels = np.array(labels)[valid]

    n = len(labels)
    print(f"Scored pairs: {n}")
    y = labels

    def score(graph_probs):
        ens_p = ens.predict(seq_probs, np.array(graph_probs))
        pred = (ens_p > 0.5).astype(int)
        return {
            "n": n, "accuracy": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, ens_p),
            "f1": f1_score(y, pred), "positive_rate": float(y.mean()),
        }

    out = {
        "description": "Simulated semi-cold-start eval: one pair endpoint physically removed from the trained "
                        "graph and reconstructed via the production insert_novel_node_knn() path; no retraining. "
                        "See scripts/cold_start_eval.py docstring for caveats before citing as inductive generalization.",
        "n_novel_proteins_removed": len(novel_proteins),
        "k_neighbors": args.k,
        "seed": args.seed,
        "cold_start_novel_protein_via_knn": score(cold_graph_probs),
        "warm_baseline_same_pairs_full_graph": score(warm_graph_probs),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
