"""
External benchmark: HuRI / HI-union (Luck et al., Nature 2020; interactome-atlas.org).

Unlike SHS27k (see external_benchmark_shs27k.py), HuRI is a genuinely INDEPENDENT-SOURCE dataset:
it comes from a systematic yeast-two-hybrid screen, not from STRING or any text-mining/curated
database this project's training data is derived from. This is the closest thing to a true
out-of-domain check available without a paid/gated dataset.

Pipeline (no retraining anywhere):
  1. HI-union.tsv (data/raw/huri/HI-union.tsv, 64,006 Ensembl-GENE-ID pairs, downloaded from
     interactome-atlas.org, no registration required for this file) is subsampled to a fixed,
     disclosed number of pairs for feasible runtime.
  2. Canonical protein sequence per gene is fetched from the public Ensembl REST API
     (rest.ensembl.org/sequence/id, batched, no API key required) and cached locally so repeated
     runs do not re-hit the network. The FIRST sequence Ensembl returns for a gene is used as its
     canonical protein (a simple, deterministic, disclosed choice -- not necessarily MANE Select).
  3. Matched negative pairs are generated the same way as external_benchmark_shs27k.py.
  4. Every fetched sequence is checked against this project's own training sequences
     (data/processed/sequences_cache.json) for a byte-identical match, exactly like the SHS27k
     script, so any residual overlap is measured rather than assumed.
  5. Scored with the existing checkpoints via the exact production inference path (warm graph
     lookup for a byte-identical match, cold-start KNN reconstruction otherwise).

Caveats:
  - Subsampled, not the full 64,006-pair set (see --n_pairs).
  - "Canonical" sequence selection is the first Ensembl API result, not a rigorously chosen
    principal isoform -- a source of possible mislabeling for a handful of genes with unusual
    isoform structures.
  - HuRI is a systematic screen with its own known false-negative rate (Y2H does not detect every
    real interaction), so a "negative" pair here means "not observed as positive in HuRI's screen",
    not "experimentally confirmed non-interacting".
"""
import sys, os, json, time, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import requests
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from src.utils.paths import PROCESSED_DATA_DIR as P, MODELS_DIR as M, PROJECT_ROOT, RAW_DATA_DIR
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.data.feature_extraction import ESMFeatureExtractor

ENSEMBL_SEQ_URL = "https://rest.ensembl.org/sequence/id"
SEQ_CACHE_PATH = RAW_DATA_DIR / "huri" / "gene_sequence_cache.json"
EMB_CACHE_PATH = RAW_DATA_DIR / "huri" / "novel_embeddings_cache.pt"


def canon(a, b):
    return (a, b) if a <= b else (b, a)


def fetch_sequences(gene_ids, batch_size=20, timeout=120, pause=0.5, max_retries=2):
    """Batched Ensembl REST lookup, gene ID -> first returned canonical-ish protein sequence.
    Results are cached to disk incrementally (after every batch) so a second run of this script,
    or a resume after a partial failure, only fetches what is still missing.

    A 50-gene batch was measured to take ~70s (Ensembl returns every isoform per gene, not just
    one), so a smaller batch size and a generous timeout are used deliberately, with retries for
    transient failures rather than silently dropping a whole batch."""
    cache = {}
    if SEQ_CACHE_PATH.exists():
        cache = json.loads(SEQ_CACHE_PATH.read_text())
    todo = [g for g in gene_ids if g not in cache]
    print(f"{len(cache)} gene sequences already cached; fetching {len(todo)} new ones from Ensembl REST...")
    n_failed_batches = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        data = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(ENSEMBL_SEQ_URL, headers={"Content-Type": "application/json", "Accept": "application/json"},
                                      json={"ids": batch, "type": "protein"}, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"  batch {i}-{i+len(batch)} attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
        if data is None:
            n_failed_batches += 1
            continue
        seen = set()
        for item in data:
            g = item.get("query")
            if g and g not in seen and "seq" in item:
                cache[g] = item["seq"]
                seen.add(g)
        print(f"  fetched {i+len(batch)}/{len(todo)} ({len(cache)} cached total)", flush=True)
        time.sleep(pause)
        SEQ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SEQ_CACHE_PATH.write_text(json.dumps(cache))
    if n_failed_batches:
        print(f"WARNING: {n_failed_batches} batches failed after retries and were skipped "
              f"(their genes are simply absent from the cache, not silently mislabeled).")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=700, help="random positive pairs to sample from HI-union")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "assets" / "evaluation" / "external_benchmark_huri.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    huri_path = RAW_DATA_DIR / "huri" / "HI-union.tsv"
    if not huri_path.exists():
        print(f"HuRI dataset not found at {huri_path}. Downloading from interactome-atlas.org...", flush=True)
        huri_path.parent.mkdir(parents=True, exist_ok=True)
        import requests
        url = "http://interactome-atlas.org/data/HI-union.tsv"
        r = requests.get(url, verify=False, stream=True, timeout=120)
        r.raise_for_status()
        with open(huri_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded HuRI HI-union.tsv ({huri_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    full = pd.read_csv(huri_path, sep="\t", header=None, names=["g1", "g2"])
    print(f"Full HI-union: {len(full)} pairs, {len(set(full.g1)|set(full.g2))} unique genes")
    sample_idx = rng.choice(len(full), size=min(args.n_pairs, len(full)), replace=False)
    sample = full.iloc[sample_idx]
    pos_pairs = set(canon(a, b) for a, b in zip(sample.g1, sample.g2))
    print(f"Sampled {len(pos_pairs)} unique positive gene pairs")

    genes = sorted(set(g for p in pos_pairs for g in p))
    seq_cache = fetch_sequences(genes)
    genes = [g for g in genes if g in seq_cache and len(seq_cache[g]) >= 20]
    pos_pairs = set(p for p in pos_pairs if p[0] in seq_cache and p[1] in seq_cache
                     and len(seq_cache[p[0]]) >= 20 and len(seq_cache[p[1]]) >= 20)
    print(f"Genes with a usable sequence: {len(genes)}; positive pairs retained: {len(pos_pairs)}")

    gene_arr = np.array(genes)
    neg_pairs = set()
    attempts, target = 0, len(pos_pairs)
    while len(neg_pairs) < target and attempts < target * 50:
        a, b = rng.choice(gene_arr, 2, replace=False)
        c = canon(a, b)
        if c not in pos_pairs and c not in neg_pairs:
            neg_pairs.add(c)
        attempts += 1
    print(f"Generated {len(neg_pairs)} negative pairs")

    emb = {k_: (v.float() if v.dtype == torch.float16 else v) for k_, v in torch.load(P / "embeddings.pt", weights_only=False).items()}
    seqs_cache = json.load(open(P / "sequences_cache.json"))
    rev = {}
    for pid, s in seqs_cache.items():
        if pid in emb:
            rev.setdefault(s, pid)
    gene_to_id, matched, novel = {}, 0, 0
    for g in genes:
        s = seq_cache[g]
        if s in rev:
            gene_to_id[g] = rev[s]
            matched += 1
        else:
            gene_to_id[g] = f"HuRI_{g}"
            novel += 1
    print(f"Matched to existing training protein (byte-identical sequence): {matched}; novel: {novel}")

    novel_seqs = {gene_to_id[g]: seq_cache[g] for g in genes if seq_cache[g] not in rev}
    emb_cache = torch.load(EMB_CACHE_PATH, weights_only=False) if EMB_CACHE_PATH.exists() else {}
    still_missing = {k_: v for k_, v in novel_seqs.items() if k_ not in emb_cache}
    if still_missing:
        print(f"Extracting ESM-2 embeddings for {len(still_missing)} novel sequences on {dev}...")
        extractor = ESMFeatureExtractor(device=str(dev))
        new_emb = extractor.get_embeddings(still_missing, batch_size=32 if dev.type == "cuda" else 4)
        emb_cache.update(new_emb)
        EMB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb_cache, EMB_CACHE_PATH)
    emb.update({k_: emb_cache[k_] for k_ in novel_seqs})

    all_pairs = [(a, b, 1) for a, b in pos_pairs] + [(a, b, 0) for a, b in neg_pairs]
    rows = [(gene_to_id[a], gene_to_id[b], label, (seq_cache[a] in rev) and (seq_cache[b] in rev)) for a, b, label in all_pairs]
    bench_df = pd.DataFrame(rows, columns=["protein1", "protein2", "label", "both_seen_in_training"])
    print(f"Benchmark set: {len(bench_df)} pairs ({bench_df.label.mean():.3f} positive rate)")

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
        if i % 500 == 0:
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
        tn, fp, fn, tp = confusion_matrix(yy, prr).ravel()
        return {"n": int(mask.sum()), "accuracy": accuracy_score(yy, prr), "roc_auc": roc_auc_score(yy, pp),
                "f1": f1_score(yy, prr), "positive_rate": float(yy.mean()),
                "predicted_positive_rate": float(prr.mean()), "mean_predicted_prob": float(pp.mean()),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}

    both_seen = bench_df["both_seen_in_training"].values
    out = {
        "description": "HuRI / HI-union (Luck et al. 2020, yeast-two-hybrid) -- a genuinely "
                        "independent-source benchmark (not derived from STRING). Subsampled and "
                        "scored with the existing checkpoints via the exact production inference "
                        "path; no retraining. See this script's module docstring for full caveats.",
        "n_pairs_sampled_from_full_huri": args.n_pairs,
        "n_total_pairs_scored": len(bench_df),
        "n_unique_genes": len(genes),
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
