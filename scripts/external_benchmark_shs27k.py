"""
External benchmark: SHS27k (Chen et al., Bioinformatics 2019), already present at
data/raw/data/SHS27k/data/train-00000-of-00001.parquet (a HuggingFace mirror, positive pairs only,
labeled by interaction TYPE, not binary interact/non-interact).

IMPORTANT SCOPE NOTE (read before quoting these numbers):
SHS27k is itself curated from STRING -- the same underlying database this project's training data
comes from, just a different snapshot/filtering (Chen et al.'s high-confidence, sequence-length-
filtered subset). It is NOT an independent-source dataset the way HuRI (yeast-two-hybrid) would be.
A direct sequence check found 1376 of SHS27k's 1690 unique proteins (81.4%) are byte-identical to
proteins already in this project's training data. This script therefore reports two numbers:
  1. "overall": every SHS27k pair, run through the same production code path as /predict (warm
     graph lookup for proteins with a matching training-time node; cold-start KNN reconstruction
     -- see scripts/cold_start_eval.py -- for the ~19% that are not byte-identical to any training
     protein).
  2. "novel_subset_only": restricted to pairs where AT LEAST ONE protein is not byte-identical to
     any training protein -- the closest this dataset can offer to an out-of-training-set check.
Report SHS27k as "a different curation/snapshot of the same source database", which is standard
practice for this benchmark in the PPI-GNN literature (PIPR, GNN-PPI, etc. use it the same way),
not as evidence of generalization to a fully independent data source.

No retraining happens anywhere in this script; it only runs inference with the existing checkpoints.
"""
import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from src.utils.paths import PROCESSED_DATA_DIR as P, MODELS_DIR as M, PROJECT_ROOT, RAW_DATA_DIR
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.data.feature_extraction import ESMFeatureExtractor


def canon(a, b):
    return (a, b) if a <= b else (b, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "assets" / "evaluation" / "external_benchmark_shs27k.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    shs_path = RAW_DATA_DIR / "data" / "SHS27k" / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(shs_path)
    print(f"Loaded SHS27k: {len(df)} rows")

    pos_pairs = set(canon(a, b) for a, b in zip(df.SeqA, df.SeqB) if a != b)
    seqs = sorted(set(df.SeqA) | set(df.SeqB))
    print(f"Unique sequences: {len(seqs)}, unique canonical positive pairs: {len(pos_pairs)}")

    # negatives: random pairs among the same protein pool, excluding known positives, same count
    seq_arr = np.array(seqs)
    neg_pairs = set()
    attempts = 0
    target = len(pos_pairs)
    while len(neg_pairs) < target and attempts < target * 50:
        a, b = rng.choice(seq_arr, 2, replace=False)
        c = canon(a, b)
        if c not in pos_pairs and c not in neg_pairs:
            neg_pairs.add(c)
        attempts += 1
    print(f"Generated {len(neg_pairs)} negative pairs (target {target})")

    # embeddings: reuse the project's cached embeddings for matched proteins, extract fresh ones
    # (CPU) only for the genuinely novel sequences
    emb = {k_: (v.float() if v.dtype == torch.float16 else v) for k_, v in torch.load(P / "embeddings.pt", weights_only=False).items()}

    # map sequence -> real ENSP id if byte-identical to a training protein AND that id actually has
    # a cached embedding (sequences_cache.json and embeddings.pt are not perfectly in sync -- a
    # handful of cached sequences have no embedding), else a synthetic id + fresh extraction
    seqs_cache = json.load(open(P / "sequences_cache.json"))
    rev = {}
    for pid, s in seqs_cache.items():
        if pid in emb:
            rev.setdefault(s, pid)
    seq_to_id, matched, novel = {}, 0, 0
    for i, s in enumerate(seqs):
        if s in rev:
            seq_to_id[s] = rev[s]
            matched += 1
        else:
            seq_to_id[s] = f"SHS27k_novel_{i:05d}"
            novel += 1
    print(f"Matched to existing training protein (byte-identical sequence, with embedding): {matched}; novel: {novel}")
    novel_seqs = {seq_to_id[s]: s for s in seqs if s not in rev}
    if novel_seqs:
        print(f"Extracting ESM-2 embeddings for {len(novel_seqs)} novel sequences on {dev}...")
        extractor = ESMFeatureExtractor(device=str(dev))
        new_emb = extractor.get_embeddings(novel_seqs, batch_size=32 if dev.type == "cuda" else 4)
        emb.update(new_emb)
    all_pairs = [(canon(a, b), 1) for a, b in pos_pairs] + [(canon(a, b), 0) for a, b in neg_pairs]
    rows = [(seq_to_id[a], seq_to_id[b], label, (a in rev) and (b in rev)) for (a, b), label in all_pairs]
    bench_df = pd.DataFrame(rows, columns=["protein1", "protein2", "label", "both_seen_in_training"])
    print(f"Benchmark set: {len(bench_df)} pairs ({bench_df.label.mean():.3f} positive rate)")

    # models
    mapping = torch.load(P / "ppi_graph_mapping.pt", weights_only=False)
    graph = torch.load(P / "ppi_graph.pt", weights_only=False).to(dev)
    esm_dim = graph.x.shape[1] - 3
    sm = SequencePPIModel(input_dim=esm_dim, hidden_dim=1024)
    sm.load_state_dict(torch.load(M / "sequence_model_best.pth", map_location=dev))
    sm.to(dev).eval()
    gm = SAGELinkPredictor(in_channels=graph.x.shape[1], hidden_channels=256)
    gm.load_state_dict(torch.load(M / "graph_model_best.pth", map_location=dev, weights_only=False))
    gm.to(dev).eval()
    ens = PPIEnsemble(str(M / "ensemble_model.pkl"))

    existing_embeddings = {pid: graph.x[idx, :esm_dim] for pid, idx in mapping.items()}
    mat = torch.stack([existing_embeddings[i] for i in existing_embeddings])
    ids_list = list(existing_embeddings.keys())
    mat_norm = F.normalize(mat.float(), dim=1)

    def knn(novel_emb, k):
        q = F.normalize(novel_emb.float().to(dev).unsqueeze(0), dim=1)
        sims = (mat_norm @ q.T).squeeze(1)
        top = torch.topk(sims, k=min(k, len(ids_list))).indices.tolist()
        return [ids_list[i] for i in top]

    with torch.no_grad():
        z_full = gm.encode(graph.x, graph.edge_index)

    print("Scoring pairs...", flush=True)
    e1b = torch.stack([emb[p] for p in bench_df.protein1]).float().to(dev)
    e2b = torch.stack([emb[p] for p in bench_df.protein2]).float().to(dev)
    with torch.no_grad():
        seq_probs = torch.sigmoid(sm(e1b, e2b)).squeeze(-1).cpu().numpy()

    graph_probs = []
    for i, (p1, p2) in enumerate(zip(bench_df.protein1, bench_df.protein2)):
        if i % 1000 == 0:
            print(f"  graph scoring {i}/{len(bench_df)}", flush=True)
        idx1_list = [mapping[p1]] if p1 in mapping else [mapping[n] for n in knn(emb[p1], args.k)]
        idx2_list = [mapping[p2]] if p2 in mapping else [mapping[n] for n in knn(emb[p2], args.k)]
        src, dst = [], []
        for i1 in idx1_list:
            for i2 in idx2_list:
                src.append(i1); dst.append(i2)
        with torch.no_grad():
            g_out = gm.decode(z_full, torch.tensor(src, device=dev), torch.tensor(dst, device=dev))
            graph_probs.append(torch.sigmoid(g_out).mean().item())
    graph_probs = np.array(graph_probs)

    ens_p = ens.predict(seq_probs, graph_probs)
    pred = (ens_p > 0.5).astype(int)
    y = bench_df.label.values

    def score(mask):
        yy, pp, prr = y[mask], ens_p[mask], pred[mask]
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            return {"n": int(mask.sum()), "note": "insufficient class diversity"}
        return {"n": int(mask.sum()), "accuracy": accuracy_score(yy, prr), "roc_auc": roc_auc_score(yy, pp),
                "f1": f1_score(yy, prr), "positive_rate": float(yy.mean())}

    both_seen = bench_df["both_seen_in_training"].values
    out = {
        "description": "SHS27k (Chen et al.) positive interactions + generated negatives, scored with "
                        "the existing checkpoints via the exact production inference path (warm graph "
                        "lookup or cold-start KNN reconstruction as appropriate). SHS27k is drawn from "
                        "the same underlying database (STRING) as this project's training data, just a "
                        "different curated snapshot -- NOT an independent-source benchmark like HuRI. See "
                        "this script's module docstring for the full caveat before citing these numbers.",
        "n_total_pairs": len(bench_df),
        "n_unique_proteins": len(seqs),
        "n_proteins_byte_identical_to_training": matched,
        "n_proteins_novel": novel,
        "overall": score(np.ones(len(y), dtype=bool)),
        "both_proteins_seen_in_training_subset": score(both_seen),
        "at_least_one_novel_protein_subset": score(~both_seen),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
