import numpy as np
import torch
import pandas as pd
import argparse
import hashlib
import os
import sys
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import SAGELinkPredictor, GINLinkPredictor
from src.models.ensemble_model import PPIEnsemble
from src.training.base_trainers import (
    SEQ_CFG, GRAPH_CFG, get_device, build_embedding_table, fit_sequence, fit_graph, graph_logits, sequence_probs,
)
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, CHECKPOINT_DIR
from src.utils.bio_encoder import BioFeatureEncoder
from src.utils.calibration import PlattScaler, calibration_report
from src.utils.seed import set_seed
from src.utils.topo_features import node_topology_columns


def load_base_models(seq_model_path, graph_model_path, graph_data, input_dim, in_channels, device):
    """Loads full-dataset base Sequence and Graph models from checkpoints (used ONLY for final validation evaluation)."""
    seq_model = SequencePPIModel(input_dim=input_dim).to(device)
    if os.path.exists(seq_model_path):
        seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
        print(f"Loaded Final Sequence Model from {seq_model_path}")
    else:
        raise FileNotFoundError(f"Sequence model checkpoint not found at {seq_model_path}")
    seq_model.eval()

    state_dict = torch.load(graph_model_path, map_location=device)
    is_gin = any("convs" in k for k in state_dict.keys())
    if is_gin:
        print("Detected GIN architecture for Graph Model.")
        graph_model = GINLinkPredictor(in_channels=in_channels, hidden_channels=128).to(device)
    else:
        print("Detected GraphSAGE architecture for Graph Model.")
        graph_model = SAGELinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)

    graph_model.load_state_dict(state_dict)
    print(f"Loaded Final Graph Model from {graph_model_path}")
    graph_model.eval()

    return seq_model, graph_model


def predict_sequence_model(model, embeddings, bio_mapping, p1_list, p2_list, device, bio_dim=0, batch_size=64):
    """Runs batch inference for Sequence Model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(p1_list), batch_size):
            p1_batch = p1_list[i:i+batch_size]
            p2_batch = p2_list[i:i+batch_size]

            b_emb1, b_emb2 = [], []
            for p1, p2 in zip(p1_batch, p2_batch):
                e1 = embeddings[p1].float()
                e2 = embeddings[p2].float()
                e1_v = e1.mean(dim=0) if e1.dim() > 1 else e1
                e2_v = e2.mean(dim=0) if e2.dim() > 1 else e2
                if bio_mapping:
                    b1 = bio_mapping.get(p1, torch.zeros(bio_dim))
                    b2 = bio_mapping.get(p2, torch.zeros(bio_dim))
                    e1_v = torch.cat([e1_v, b1])
                    e2_v = torch.cat([e2_v, b2])
                b_emb1.append(e1_v)
                b_emb2.append(e2_v)

            e1_t = torch.stack(b_emb1).to(device)
            e2_t = torch.stack(b_emb2).to(device)
            out = model(e1_t, e2_t)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds.extend(probs)
    return np.array(preds)


def predict_graph_model(graph_model, fold_graph_data, node_mapping, p1_list, p2_list, device, chunk_size=1000):
    """Runs batch inference for Graph Model."""
    graph_model.eval()
    src_indices = [node_mapping[p1] for p1 in p1_list]
    dst_indices = [node_mapping[p2] for p2 in p2_list]
    edge_label_index = torch.tensor([src_indices, dst_indices], dtype=torch.long).to(device)

    preds = []
    with torch.no_grad():
        z = graph_model.encode(fold_graph_data.x.to(device), fold_graph_data.edge_index.to(device))
        num_edges = edge_label_index.size(1)
        for i in range(0, num_edges, chunk_size):
            chunk = edge_label_index[:, i:i+chunk_size]
            out = graph_model.decode(z, chunk[0], chunk[1])
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds.extend(probs)
    return np.array(preds)


def _fingerprint(df):
    h = hashlib.md5()
    h.update("|".join(df["protein1"].astype(str)).encode())
    h.update("|".join(df["protein2"].astype(str)).encode())
    h.update(df["label"].values.tobytes())
    return h.hexdigest()


def generate_oof_predictions(
    train_df,
    embeddings,
    node_mapping,
    full_graph_data,
    device,
    k_folds=5,
    es_frac=0.1,
    seed=42,
    checkpoint_dir=None,
    ckpt_every=5,
    seq_cfg=SEQ_CFG,
    graph_cfg=GRAPH_CFG,
    dry_run=False,
):
    """
    Generates leakage-free out-of-fold (OOF) predictions on train.csv.

    Per fold, everything a base model sees is derived from the fold's TRAINING rows only:
      * the message-passing graph holds only in-fold positive pairs, and its node columns
        (degree centrality, clustering coefficient, PageRank) are RECOMPUTED on that graph;
      * models are trained with the same max-epoch budget / patience / loss / scheduler as the production
        models (base_trainers.SEQ_CFG / GRAPH_CFG). Early stopping uses an inner validation split carved
        from the fold's training rows (es_frac), never the held-out rows.
    Completed folds are cached in checkpoint_dir so an interrupted Colab session resumes at the next fold.

    GraphSAGE calibration: per fold, a Platt scaler is fit on the fold's inner early-stopping logits (never on the
    held-out rows) and applied to the held-out logits, mirroring how the production calibrator is fit on val.csv.
    Returns (oof_seq, oof_graph_raw, oof_graph_calibrated, visited).
    """
    print(f"\n--- Leakage-free OOF predictions: {k_folds} folds, max {seq_cfg['max_epochs']}/{graph_cfg['max_epochs']} "
          f"epochs (seq/graph) with early stopping, device={device} ---")

    n_samples = len(train_df)
    oof_seq = np.zeros(n_samples, dtype=np.float32)
    oof_graph = np.zeros(n_samples, dtype=np.float32)
    oof_graph_logit = np.zeros(n_samples, dtype=np.float32)
    oof_graph_cal = np.zeros(n_samples, dtype=np.float32)
    visited = np.zeros(n_samples, dtype=bool)

    p1_all, p2_all = train_df["protein1"].values, train_df["protein2"].values
    labels_all = train_df["label"].values
    fp = _fingerprint(train_df)
    oof_dir = os.path.join(checkpoint_dir, "oof") if checkpoint_dir else None
    if oof_dir:
        os.makedirs(oof_dir, exist_ok=True)

    protein_list = sorted(set(p1_all) | set(p2_all))
    table, row_of = build_embedding_table(embeddings, protein_list, device)
    seq_a = np.array([row_of[p] for p in p1_all])
    seq_b = np.array([row_of[p] for p in p2_all])
    g_u = np.array([node_mapping[p] for p in p1_all])
    g_v = np.array([node_mapping[p] for p in p2_all])

    base_x = full_graph_data.x.cpu()
    num_nodes = base_x.shape[0]
    in_channels = base_x.shape[1]
    input_dim = table.shape[1]
    # the cache key must change with the embedding model too, or folds computed on old embeddings would be reused
    fp = f"{fp}-esm{input_dim}-x{in_channels}"

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_times = []
    for fold, (train_idx, ho_idx) in enumerate(skf.split(train_df, labels_all)):
        cache_file = os.path.join(oof_dir, f"fold{fold + 1}.npz") if oof_dir else None
        if cache_file and os.path.exists(cache_file) and not dry_run:
            c = np.load(cache_file, allow_pickle=False)
            if str(c["fingerprint"]) == fp and np.array_equal(c["ho_idx"], ho_idx):
                oof_seq[ho_idx], oof_graph[ho_idx], oof_graph_logit[ho_idx] = c["seq"], c["graph"], c["graph_logit"]
                oof_graph_cal[ho_idx] = c["graph_cal"]
                visited[ho_idx] = True
                print(f"\nFold {fold + 1}/{k_folds}: loaded finished result from {cache_file}")
                continue

        t_fold = time.time()
        set_seed(seed + fold)
        print(f"\nFold {fold + 1}/{k_folds} (in-fold train: {len(train_idx)}, held-out OOF: {len(ho_idx)})")
        assert set(train_idx).isdisjoint(set(ho_idx)), f"Fold {fold + 1}: train and held-out indices overlap!"

        # inner split of the fold's training rows: fit / early-stopping validation
        fit_idx, es_idx = train_test_split(train_idx, test_size=es_frac, stratify=labels_all[train_idx],
                                           random_state=seed + fold)

        # fold graph = positive FIT pairs only; ES and held-out pairs must not be message-passing edges
        pos_fit = fit_idx[labels_all[fit_idx] == 1]
        f_src, f_dst = g_u[pos_fit].tolist(), g_v[pos_fit].tolist()
        fold_edges = set(zip(f_src, f_dst)) | set(zip(f_dst, f_src))
        for name, idx in (("held-out", ho_idx), ("early-stop", es_idx)):
            pairs = set(zip(g_u[idx].tolist(), g_v[idx].tolist())) | set(zip(g_v[idx].tolist(), g_u[idx].tolist()))
            leaked = pairs & fold_edges
            assert not leaked, f"LEAKAGE in fold {fold + 1}: {len(leaked)} {name} pairs are edges of the fold graph!"
        print(f"  [VERIFIED] no held-out / early-stop pair is an edge of the fold graph ({len(pos_fit)} positive edges).")

        # per-fold node columns, computed ONLY from this fold's edges (never copied from the full graph)
        topo = node_topology_columns(f_src, f_dst, num_nodes)
        x_fold = torch.cat([base_x[:, :-3], topo], dim=-1)
        assert x_fold.shape[1] == in_channels
        assert not torch.equal(x_fold[:, -3:], base_x[:, -3:]), "fold node features equal the full-graph features"
        ei_fold = torch.tensor([f_src + f_dst, f_dst + f_src], dtype=torch.long)

        print("  Training sequence model...")
        seq_m = SequencePPIModel(input_dim=input_dim).to(device)
        fit_sequence(seq_m, table, seq_a[fit_idx], seq_b[fit_idx], labels_all[fit_idx],
                     seq_a[es_idx], seq_b[es_idx], labels_all[es_idx], device, cfg=seq_cfg,
                     ckpt_path=os.path.join(checkpoint_dir, f"seq_fold{fold + 1}.pt") if checkpoint_dir else None,
                     ckpt_every=ckpt_every, tag=f"seq f{fold + 1}")

        print("  Training GraphSAGE model...")
        gnn_m = SAGELinkPredictor(in_channels=in_channels, hidden_channels=256).to(device)
        fit_graph(gnn_m, x_fold, ei_fold, g_u[fit_idx], g_v[fit_idx], labels_all[fit_idx],
                  g_u[es_idx], g_v[es_idx], labels_all[es_idx], device, cfg=graph_cfg,
                  ckpt_path=os.path.join(checkpoint_dir, f"graph_fold{fold + 1}.pt") if checkpoint_dir else None,
                  ckpt_every=ckpt_every, tag=f"graph f{fold + 1}")

        oof_seq[ho_idx] = sequence_probs(seq_m, table, seq_a[ho_idx], seq_b[ho_idx], device)
        logits = graph_logits(gnn_m, x_fold, ei_fold, g_u[ho_idx], g_v[ho_idx], device)
        oof_graph_logit[ho_idx] = logits
        oof_graph[ho_idx] = 1.0 / (1.0 + np.exp(-logits))
        es_logits = graph_logits(gnn_m, x_fold, ei_fold, g_u[es_idx], g_v[es_idx], device)
        fold_cal = PlattScaler().fit(es_logits, labels_all[es_idx])
        oof_graph_cal[ho_idx] = fold_cal.transform_logits(logits)
        print(f"  Fold {fold + 1} Platt scaler (fit on early-stop split): a={fold_cal.a:.3f} b={fold_cal.b:.3f}")
        visited[ho_idx] = True

        print(f"  Fold {fold + 1} held-out AUC: seq {roc_auc_score(labels_all[ho_idx], oof_seq[ho_idx]):.4f} | "
              f"graph {roc_auc_score(labels_all[ho_idx], oof_graph[ho_idx]):.4f}")

        if cache_file and not dry_run:
            np.savez(cache_file, fingerprint=fp, ho_idx=ho_idx, seq=oof_seq[ho_idx], graph=oof_graph[ho_idx],
                     graph_logit=oof_graph_logit[ho_idx], graph_cal=oof_graph_cal[ho_idx])
        fold_times.append(time.time() - t_fold)
        remaining = (k_folds - fold - 1) * float(np.mean(fold_times))
        print(f"  Fold {fold + 1} took {fold_times[-1] / 60:.1f} min | est. remaining {remaining / 60:.1f} min")

        del seq_m, gnn_m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if dry_run:
            print("[DRY-RUN] Stopping after fold 1.")
            break

    if not dry_run:
        assert visited.all(), "Not all train.csv samples received an OOF prediction!"
        print(f"\n[VERIFIED] All {n_samples} training samples received exactly one leakage-free OOF prediction.")
    return oof_seq, oof_graph, oof_graph_cal, visited


def train_ensemble(seq_model_path, graph_model_path, graph_data_path, k_folds=5, dry_run=False, max_samples=None,
                   seed=42, checkpoint_dir=None, ckpt_every=5, force_cpu=False, es_frac=0.1):
    device = get_device(force_cpu)
    print(f"Running Ensemble Training Pipeline on {device}...")
    t_start = time.time()

    # 1. Load Support Files
    print("Loading support files (embeddings, graph data, node mapping, bio features)...")
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()

    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    if not emb_path.exists() or not map_path.exists() or not os.path.exists(graph_data_path):
        raise FileNotFoundError("Required embeddings, node mapping, or graph data file not found.")

    embeddings = torch.load(emb_path, weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    node_mapping = torch.load(map_path, weights_only=False)
    full_graph_data = torch.load(graph_data_path, weights_only=False)  # stays on CPU; moved to `device` where used

    # 2. Load Train Dataset for Meta-Learner Training
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)

    # Filter train_df to valid entries present in mapping & embeddings
    filtered_train_df = train_df[
        train_df["protein1"].isin(embeddings) &
        train_df["protein2"].isin(embeddings) &
        train_df["protein1"].isin(node_mapping) &
        train_df["protein2"].isin(node_mapping)
    ].copy().reset_index(drop=True)

    if dry_run and max_samples:
        print(f"[DRY-RUN] Subsampling train data to {max_samples} samples...")
        filtered_train_df = filtered_train_df.iloc[:max_samples].copy().reset_index(drop=True)

    print(f"Meta-Learner Training Dataset: {len(filtered_train_df)} samples from train.csv.")

    # 3. Generate Leakage-Free OOF Base-Model Predictions on train.csv
    seq_cfg, graph_cfg = dict(SEQ_CFG), dict(GRAPH_CFG)
    if dry_run:  # structure/leakage check only: 2 epochs per model
        seq_cfg["max_epochs"] = graph_cfg["max_epochs"] = 2
    oof_seq_preds, oof_graph_raw, oof_graph_cal, visited_indices = generate_oof_predictions(
        filtered_train_df,
        embeddings,
        node_mapping,
        full_graph_data,
        device,
        k_folds=k_folds,
        es_frac=es_frac,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        ckpt_every=ckpt_every,
        seq_cfg=seq_cfg,
        graph_cfg=graph_cfg,
        dry_run=dry_run,
    )

    if dry_run:
        print("\n[DRY-RUN VERIFICATION COMPLETE] OOF pipeline structure and leakage assertions passed successfully.")
        return

    train_labels_np = filtered_train_df["label"].values

    # 4. GraphSAGE calibration on the OOF predictions (raw vs per-fold Platt-calibrated)
    rep = calibration_report(train_labels_np, oof_graph_raw, oof_graph_cal)
    print(f"OOF GraphSAGE calibration  before: {rep['before']}  after: {rep['after']}")
    print(f"OOF AUC  seq {roc_auc_score(train_labels_np, oof_seq_preds):.4f} | graph {roc_auc_score(train_labels_np, oof_graph_raw):.4f}")
    graph_cal_path = os.path.join(os.path.dirname(str(graph_model_path)), "graph_calibrator.json")
    if not os.path.exists(graph_cal_path):
        raise FileNotFoundError(f"{graph_cal_path} missing. Run train_graph_model.py first: it fits the production "
                                "GraphSAGE calibrator on val.csv, and the meta-learner must be paired with it.")
    graph_calibrator = PlattScaler.load(graph_cal_path)

    # 5. Train XGBoost Meta-Learner strictly on OOF predictions and train labels
    print("\n--- Fitting XGBoost Meta-Learner ---")
    ensemble = PPIEnsemble()
    ensemble.train_stacking(oof_seq_preds, oof_graph_cal, train_labels_np)
    ensemble.graph_calibrator = graph_calibrator  # saved next to the meta-learner; applied inside predict()

    # Save Meta-Learner Model
    out_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    os.makedirs(out_path.parent, exist_ok=True)
    ensemble.save(out_path)
    print("Ensemble meta-learner training and save complete.")

    # 6. Separate Evaluation on Held-Out Validation Set (val.csv)
    print("\n--- Evaluating Trained Ensemble on Validation Set (val.csv) ---")
    val_path = PROCESSED_DATA_DIR / "val.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        filtered_val_df = val_df[
            val_df["protein1"].isin(embeddings) &
            val_df["protein2"].isin(embeddings) &
            val_df["protein1"].isin(node_mapping) &
            val_df["protein2"].isin(node_mapping)
        ].copy().reset_index(drop=True)

        print(f"Validation Dataset: {len(filtered_val_df)} samples from val.csv.")

        # Load final trained base models for validation evaluation ONLY
        sample_emb = next(iter(embeddings.values()))
        input_dim = sample_emb.shape[-1]
        in_channels = full_graph_data.x.shape[1]
        seq_model, graph_model = load_base_models(seq_model_path, graph_model_path, full_graph_data, input_dim, in_channels, device)

        val_p1 = filtered_val_df["protein1"].values
        val_p2 = filtered_val_df["protein2"].values
        val_labels_np = filtered_val_df["label"].values

        val_seq_preds = predict_sequence_model(seq_model, embeddings, bio_mapping, val_p1, val_p2, device, bio_dim=len(next(iter(bio_mapping.values()))) if bio_mapping else 0)
        val_graph_preds = predict_graph_model(graph_model, full_graph_data, node_mapping, val_p1, val_p2, device)

        val_ensemble_preds = ensemble.predict(val_seq_preds, val_graph_preds, method="stacking")  # raw graph in, calibrated inside
        val_graph_cal = ensemble.calibrate_graph(val_graph_preds)

        acc_seq = accuracy_score(val_labels_np, (val_seq_preds > 0.5).astype(int))
        acc_graph = accuracy_score(val_labels_np, (val_graph_cal > 0.5).astype(int))
        acc_ens = accuracy_score(val_labels_np, (val_ensemble_preds > 0.5).astype(int))

        auc_seq = roc_auc_score(val_labels_np, val_seq_preds)
        auc_graph = roc_auc_score(val_labels_np, val_graph_preds)
        auc_ens = roc_auc_score(val_labels_np, val_ensemble_preds)

        print(f"\n--- Validation Set Performance (val.csv) ---")
        print(f"Sequence Model Acc: {acc_seq*100:.2f}% | ROC-AUC: {auc_seq:.4f}")
        print(f"Graph Model Acc (calibrated): {acc_graph*100:.2f}% | ROC-AUC: {auc_graph:.4f}")
        print(f"Ensemble Model Acc: {acc_ens*100:.2f}% | ROC-AUC: {auc_ens:.4f}")
    print(f"\nTotal ensemble pipeline time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ensemble Stacking Meta-Learner using OOF Predictions.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for OOF generation.")
    parser.add_argument("--es_frac", type=float, default=0.1,
                        help="Fraction of each fold's training rows held out for early stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR),
                        help="Where fold results / epoch checkpoints are written (mount Google Drive here on Colab).")
    parser.add_argument("--ckpt_every", type=int, default=5, help="Save a training checkpoint every N epochs.")
    parser.add_argument("--force-cpu", action="store_true", help="Ignore CUDA even if available.")
    parser.add_argument("--dry_run", action="store_true", help="Run fast dry-run verification (1 fold, subsampled, 2 epochs).")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum samples for dry-run verification.")
    args = parser.parse_args()
    set_seed(args.seed)

    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    graph_data = PROCESSED_DATA_DIR / "ppi_graph.pt"

    train_ensemble(
        seq_path,
        graph_path,
        graph_data,
        k_folds=args.k_folds,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        ckpt_every=args.ckpt_every,
        force_cpu=args.force_cpu,
        es_frac=args.es_frac,
    )
