"""
Stage 1 of the metric-optimization experiments: build leakage-controlled meta-features.

Reads ONLY train.csv and val.csv. test.csv is never opened here.

Outputs models/experiments/features_cache.npz containing, for train (out-of-fold) and val:
    seq, graph, bio, cn, jac, prd, label
Nothing in models/ensemble_model.pkl is read or written.

Leakage controls
  * OOF: the fold graph holds only in-fold POSITIVE train edges. The three node-topology
    columns of the GNN input (degree, clustering, PageRank; last 3 cols of x) are recomputed
    on that fold graph (the original code copied them from the full train graph).
  * Topology pair features (CN / Jaccard / PageRank-delta) for OOF rows use the fold graph;
    for val rows they use the full train graph (all train positives). Val/test edges never enter a graph.
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, MODELS_DIR
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor
from src.utils.bio_encoder import BioFeatureEncoder
from src.analysis.biological_managers import BiologicalManager, ensemble_bio_score
from src.training.train_ensemble import (
    train_fold_sequence, train_fold_gat, predict_sequence_model, predict_graph_model, load_base_models,
)

OUT_DIR = PROJECT_ROOT / "models" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "features_cache.npz"
LEAK_REPORT = OUT_DIR / "leakage_report.json"
SEED = 42
K_FOLDS = 5
OOF_EPOCHS = 5


def build_nx(src, dst, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src, dst))
    return G


def node_topo_columns(G, n):
    """Same recipe as src/data/graph_construction.py (degree centrality, clustering, PageRank a=0.85)."""
    dc = nx.degree_centrality(G)
    cl = nx.clustering(G)
    pr = nx.pagerank(G, alpha=0.85)
    cols = torch.zeros((n, 3), dtype=torch.float32)
    for i in range(n):
        cols[i, 0], cols[i, 1], cols[i, 2] = dc.get(i, 0), cl.get(i, 0), pr.get(i, 0)
    return cols, pr


def pair_topology(G, pr, u_idx, v_idx):
    """Common neighbours, Jaccard coefficient, |PageRank(u) - PageRank(v)| on graph G.
    The candidate edge (u,v) itself is never in G for held-out / val pairs (asserted by the caller)."""
    adj = {n: set(G[n]) for n in G.nodes}
    cn = np.zeros(len(u_idx), dtype=np.float32)
    jac = np.zeros(len(u_idx), dtype=np.float32)
    prd = np.zeros(len(u_idx), dtype=np.float32)
    for i, (u, v) in enumerate(zip(u_idx, v_idx)):
        nu, nv = adj[u], adj[v]
        inter = len(nu & nv)
        union = len(nu | nv)
        cn[i] = inter
        jac[i] = inter / union if union else 0.0
        prd[i] = abs(pr[u] - pr[v])
    return cn, jac, prd


def edge_set(src, dst):
    s = set(zip(src, dst))
    return s | {(b, a) for a, b in s}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    bio_dim = len(next(iter(bio_mapping.values()))) if bio_mapping else 0
    embeddings = torch.load(PROCESSED_DATA_DIR / "embeddings.pt", weights_only=False)
    embeddings = {k: v.float() for k, v in embeddings.items()}
    node_mapping = torch.load(PROCESSED_DATA_DIR / "ppi_graph_mapping.pt", weights_only=False)
    full_graph = torch.load(PROCESSED_DATA_DIR / "ppi_graph.pt", weights_only=False)
    n_nodes = full_graph.x.shape[0]
    in_channels = full_graph.x.shape[1]
    input_dim = next(iter(embeddings.values())).shape[-1]

    def load_split(name):
        df = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
        keep = (df.protein1.isin(embeddings) & df.protein2.isin(embeddings)
                & df.protein1.isin(node_mapping) & df.protein2.isin(node_mapping))
        return df[keep].reset_index(drop=True)

    train_df, val_df = load_split("train"), load_split("val")
    print(f"train={len(train_df)} val={len(val_df)} nodes={n_nodes} bio_dim={bio_dim}", flush=True)

    tr_u = np.array([node_mapping[p] for p in train_df.protein1])
    tr_v = np.array([node_mapping[p] for p in train_df.protein2])
    y_tr = train_df.label.values
    p1_all, p2_all = train_df.protein1.values, train_df.protein2.values

    oof = {k: np.zeros(len(train_df), dtype=np.float32) for k in ["seq", "graph", "cn", "jac", "prd"]}
    leak = {"folds": [], "notes": []}

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(train_df, y_tr)):
        tf = time.time()
        assert set(tr_idx).isdisjoint(ho_idx)
        pos_in = tr_idx[y_tr[tr_idx] == 1]
        fsrc, fdst = tr_u[pos_in].tolist(), tr_v[pos_in].tolist()
        fold_edges = edge_set(fsrc, fdst)
        ho_pairs = set(zip(tr_u[ho_idx].tolist(), tr_v[ho_idx].tolist()))
        ho_pairs |= {(b, a) for a, b in ho_pairs}
        # a held-out pair could also appear in-fold only if the dataset had duplicate/reversed pairs
        overlap = len(fold_edges & ho_pairs)
        assert overlap == 0, f"fold {fold}: {overlap} held-out pairs present in fold graph"

        G = build_nx(fsrc, fdst, n_nodes)
        topo_cols, pr = node_topo_columns(G, n_nodes)
        x_fold = torch.cat([full_graph.x[:, :-3], topo_cols], dim=-1)
        fold_edge_index = torch.tensor([fsrc + fdst, fdst + fsrc], dtype=torch.long)
        fold_graph = Data(x=x_fold, edge_index=fold_edge_index).to(device)

        seq_m = SequencePPIModel(input_dim=input_dim).to(device)
        gnn_m = SAGELinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
        train_fold_sequence(seq_m, embeddings, bio_mapping, p1_all[tr_idx], p2_all[tr_idx], y_tr[tr_idx],
                            device, bio_dim=bio_dim, epochs=OOF_EPOCHS)
        train_fold_gat(gnn_m, fold_graph, p1_all[tr_idx], p2_all[tr_idx], y_tr[tr_idx], node_mapping,
                       device, epochs=OOF_EPOCHS)
        oof["seq"][ho_idx] = predict_sequence_model(seq_m, embeddings, bio_mapping, p1_all[ho_idx], p2_all[ho_idx], device, bio_dim=bio_dim)
        oof["graph"][ho_idx] = predict_graph_model(gnn_m, fold_graph, node_mapping, p1_all[ho_idx], p2_all[ho_idx], device)
        cn, jac, prd = pair_topology(G, pr, tr_u[ho_idx], tr_v[ho_idx])
        oof["cn"][ho_idx], oof["jac"][ho_idx], oof["prd"][ho_idx] = cn, jac, prd

        leak["folds"].append({
            "fold": fold + 1, "in_fold_pos_edges": int(len(pos_in)), "held_out_rows": int(len(ho_idx)),
            "held_out_pairs_in_fold_graph": overlap,
            "topology_features_from": "fold graph (in-fold positive train edges only)",
            "gnn_node_topo_columns_from": "fold graph (recomputed)",
        })
        print(f"fold {fold+1}/{K_FOLDS} done in {time.time()-tf:.0f}s", flush=True)

    # ---- validation features: full base models, full TRAIN graph ----
    print("Computing validation features...", flush=True)
    train_pos = y_tr == 1
    tsrc, tdst = tr_u[train_pos].tolist(), tr_v[train_pos].tolist()
    train_edges = edge_set(tsrc, tdst)
    va_u = np.array([node_mapping[p] for p in val_df.protein1])
    va_v = np.array([node_mapping[p] for p in val_df.protein2])
    y_va = val_df.label.values
    val_pairs = set(zip(va_u.tolist(), va_v.tolist()))
    val_pairs |= {(b, a) for a, b in val_pairs}
    val_in_train_graph = len(train_edges & val_pairs)
    assert val_in_train_graph == 0, "val pairs present in train graph"
    # the saved ppi_graph.pt must equal the train-positive graph (i.e. contains no val/test edges)
    saved_edges = set(zip(full_graph.edge_index[0].tolist(), full_graph.edge_index[1].tolist()))
    assert saved_edges == train_edges, "ppi_graph.pt edge set != train positive edges"
    leak["val"] = {"val_pairs_in_train_graph": val_in_train_graph,
                   "saved_graph_equals_train_positive_edges": True,
                   "train_val_pair_overlap": int(len(set(zip(p1_all, p2_all)) & set(zip(val_df.protein1, val_df.protein2))))}

    Gf = build_nx(tsrc, tdst, n_nodes)
    _, prf = node_topo_columns(Gf, n_nodes)
    va_cn, va_jac, va_prd = pair_topology(Gf, prf, va_u, va_v)

    full_graph = full_graph.to(device)
    seq_m, gnn_m = load_base_models(MODELS_DIR / "sequence_model_best.pth", MODELS_DIR / "graph_model_best.pth",
                                    full_graph, input_dim, in_channels, device)
    va_seq = predict_sequence_model(seq_m, embeddings, bio_mapping, val_df.protein1.values, val_df.protein2.values, device, bio_dim=bio_dim)
    va_graph = predict_graph_model(gnn_m, full_graph, node_mapping, val_df.protein1.values, val_df.protein2.values, device)

    print("Bio scores...", flush=True)
    bm = BiologicalManager()
    tr_bio = np.array([ensemble_bio_score(bm, a, b) for a, b in zip(train_df.protein1, train_df.protein2)], dtype=np.float32)
    va_bio = np.array([ensemble_bio_score(bm, a, b) for a, b in zip(val_df.protein1, val_df.protein2)], dtype=np.float32)

    # sanity: single-feature AUC on OOF vs val. A large OOF>>val gap would indicate leakage.
    aucs = {}
    for k, o, v in [("cn", oof["cn"], va_cn), ("jac", oof["jac"], va_jac), ("prd", oof["prd"], va_prd),
                    ("seq", oof["seq"], va_seq), ("graph", oof["graph"], va_graph)]:
        aucs[k] = {"oof_auc": float(roc_auc_score(y_tr, o)), "val_auc": float(roc_auc_score(y_va, v))}
    leak["single_feature_auc_oof_vs_val"] = aucs
    print(json.dumps(aucs, indent=1), flush=True)

    np.savez_compressed(
        CACHE,
        tr_seq=oof["seq"], tr_graph=oof["graph"], tr_bio=tr_bio, tr_cn=oof["cn"], tr_jac=oof["jac"], tr_prd=oof["prd"], tr_y=y_tr,
        va_seq=va_seq, va_graph=va_graph, va_bio=va_bio, va_cn=va_cn, va_jac=va_jac, va_prd=va_prd, va_y=y_va,
    )
    leak["config"] = {"k_folds": K_FOLDS, "oof_epochs": OOF_EPOCHS, "seed": SEED, "device": str(device)}
    LEAK_REPORT.write_text(json.dumps(leak, indent=2))
    print(f"Saved {CACHE} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
