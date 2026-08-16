import os
import sys
import glob
import random
import numpy as np
import torch
from pathlib import Path
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, f1_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.irlm_module import InteractionRegionLocalizationModule
from src.training.train_irlm import IRLMDataset


# ==============================================================================
# Normalization Functions
# ==============================================================================

def norm_raw(M: np.ndarray) -> np.ndarray:
    """Raw unnormalized predicted interaction probability matrix."""
    return M.copy()


def norm_minmax(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-complex min-max normalization."""
    m_min = np.min(M)
    m_max = np.max(M)
    if m_max - m_min < eps:
        return np.zeros_like(M)
    return (M - m_min) / (m_max - m_min + eps)


def norm_percentile(M: np.ndarray) -> np.ndarray:
    """Per-complex percentile / rank normalization in range (0, 1]."""
    ranks = rankdata(M.flatten(), method="average")
    percentiles = ranks / M.size
    return percentiles.reshape(M.shape)


def norm_row(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise min-max normalization (normalizes across candidate partners for each residue in Chain A)."""
    r_min = np.min(M, axis=1, keepdims=True)
    r_max = np.max(M, axis=1, keepdims=True)
    denom = r_max - r_min
    denom[denom < eps] = 1.0
    return (M - r_min) / denom


def norm_col(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column-wise min-max normalization (normalizes across candidate residues for each partner in Chain B)."""
    c_min = np.min(M, axis=0, keepdims=True)
    c_max = np.max(M, axis=0, keepdims=True)
    denom = c_max - c_min
    denom[denom < eps] = 1.0
    return (M - c_min) / denom


def norm_dual(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Dual (Geometric Mean of Row and Column Min-Max Normalizations)."""
    M_r = norm_row(M, eps)
    M_c = norm_col(M, eps)
    return np.sqrt(M_r * M_c)


def norm_apc(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Average Product Correction (APC) widely used in DCA/ESM contact matrices."""
    row_means = np.mean(M, axis=1, keepdims=True)  # (L_A, 1)
    col_means = np.mean(M, axis=0, keepdims=True)  # (1, L_B)
    global_mean = np.mean(M)
    apc = (row_means @ col_means) / (global_mean + eps)
    return M - apc


NORMALIZATION_METHODS = {
    "Raw": norm_raw,
    "Per-complex Min-Max": norm_minmax,
    "Per-complex Percentile": norm_percentile,
    "Row-wise": norm_row,
    "Column-wise": norm_col,
    "Dual (Row x Col)": norm_dual,
    "APC (Avg Prod Corr)": norm_apc,
}


# ==============================================================================
# Helper Evaluation Functions
# ==============================================================================

def calc_topk_metrics(probs_flat: np.ndarray, targets_flat: np.ndarray, k_list: list, total_positives: int):
    """Compute Top-K Precision, Recall, F1, and Enrichment over random."""
    gt_pos_rate = total_positives / len(targets_flat)
    sorted_idx = np.argsort(probs_flat)[::-1]
    
    results = {}
    for k in k_list:
        k_clamped = min(k, len(probs_flat))
        top_k_idx = sorted_idx[:k_clamped]
        pos_in_k = int(np.sum(targets_flat[top_k_idx] == 1))
        prec = pos_in_k / k_clamped if k_clamped > 0 else 0.0
        rec = pos_in_k / total_positives if total_positives > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
        enrichment = prec / (gt_pos_rate + 1e-12)
        results[k] = {
            "pos_in_k": pos_in_k,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "enrichment": enrichment
        }
    return results


def find_optimal_threshold(probs_flat: np.ndarray, targets_flat: np.ndarray, num_points: int = 500):
    """Find threshold that maximizes F1 score."""
    min_p, max_p = np.min(probs_flat), np.max(probs_flat)
    if min_p == max_p:
        return 0.5, 0.0, 0.0, 0.0
    
    thresholds = np.linspace(min_p, max_p, num_points)
    best_f1 = -1.0
    best_tau = min_p
    best_prec = 0.0
    best_rec = 0.0

    for tau in thresholds:
        preds = (probs_flat >= tau)
        tp = np.sum(preds & (targets_flat == 1))
        fp = np.sum(preds & (targets_flat == 0))
        fn = np.sum((~preds) & (targets_flat == 1))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_prec = prec
            best_rec = rec

    return best_tau, best_f1, best_prec, best_rec


def run_eval_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = PROJECT_ROOT / "models" / "irlm_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"IRLM model checkpoint not found at {model_path}")

    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load 1YCR NPZ Artifact
    npz_1ycr_path = PROJECT_ROOT / "data" / "processed" / "irlm_dataset" / "1YCR_A_B.npz"
    data_1ycr = np.load(npz_1ycr_path)
    h_a_1ycr = torch.tensor(data_1ycr["esm_embedding_a"], dtype=torch.float32).to(device)
    h_b_1ycr = torch.tensor(data_1ycr["esm_embedding_b"], dtype=torch.float32).to(device)
    gt_cmap_1ycr = data_1ycr["contact_map"]  # (85, 13)
    z_a_1ycr = torch.tensor(data_1ycr["graph_embedding_a"], dtype=torch.float32).to(device) if "graph_embedding_a" in data_1ycr else None
    z_b_1ycr = torch.tensor(data_1ycr["graph_embedding_b"], dtype=torch.float32).to(device) if "graph_embedding_b" in data_1ycr else None

    with torch.no_grad():
        _, _, raw_matrix_1ycr_tensor = model.compute_residue_importance(h_a_1ycr, h_b_1ycr, z_a_1ycr, z_b_1ycr)
        raw_matrix_1ycr = raw_matrix_1ycr_tensor.cpu().numpy()

    # Load 14 Validation Set Complexes
    data_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    seed = 42
    rng = random.Random(seed)
    shuffled_files = list(data_files)
    rng.shuffle(shuffled_files)

    val_size = max(1, int(len(shuffled_files) * 0.2))
    val_files = shuffled_files[:val_size]

    val_dataset = IRLMDataset(val_files)
    val_raw_matrices = []
    val_gt_cmaps = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            h_a = sample["h_a"].to(device)
            h_b = sample["h_b"].to(device)
            cmap = sample["contact_map"].numpy()
            z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
            z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None

            _, _, mat_tensor = model.compute_residue_importance(h_a, h_b, z_a, z_b)
            mat_np = mat_tensor.cpu().numpy()

            val_raw_matrices.append(mat_np)
            val_gt_cmaps.append(cmap)

    print(f"Successfully loaded 1YCR and {len(val_raw_matrices)} validation complexes.")

    # ==========================================================================
    # 1. EVALUATION ON 1YCR
    # ==========================================================================
    gt_1ycr_flat = gt_cmap_1ycr.flatten()
    total_pos_1ycr = int(np.sum(gt_1ycr_flat))
    topk_1ycr_list = [5, 10, 20, 28, 50]
    rec_k_list = [10, 20, 28, 50]

    raw_1ycr_flat = raw_matrix_1ycr.flatten()

    results_1ycr = {}

    print("\n" + "=" * 90)
    print("                         PART 1: 1YCR NORMALIZATION AUDIT")
    print("=" * 90)

    for name, func in NORMALIZATION_METHODS.items():
        norm_mat = func(raw_matrix_1ycr)
        norm_flat = norm_mat.flatten()

        # Spearman rank correlation with raw scores
        rho, _ = spearmanr(raw_1ycr_flat, norm_flat)
        rank_changed = "NO (Monotonic)" if np.isclose(rho, 1.0) else f"YES (Rho={rho:.4f})"

        auprc = average_precision_score(gt_1ycr_flat, norm_flat)
        roc_auc = roc_auc_score(gt_1ycr_flat, norm_flat)

        topk_res = calc_topk_metrics(norm_flat, gt_1ycr_flat, topk_1ycr_list, total_pos_1ycr)
        best_tau, best_f1, best_prec, best_rec = find_optimal_threshold(norm_flat, gt_1ycr_flat)

        results_1ycr[name] = {
            "auprc": auprc,
            "roc_auc": roc_auc,
            "rank_changed": rank_changed,
            "spearman_rho": rho,
            "topk": topk_res,
            "best_tau": best_tau,
            "best_f1": best_f1,
            "best_prec": best_prec,
            "best_rec": best_rec,
        }

    # Print 1YCR Comparison Table
    header_1ycr = f"{'Strategy':<22} | {'Rank Changed?':<16} | {'ROC-AUC':<8} | {'AUPRC':<8} | {'P@5':<6} | {'P@10':<6} | {'P@28':<6} | {'P@50':<6} | {'R@28':<6} | {'Enrich@28':<9}"
    print(header_1ycr)
    print("-" * len(header_1ycr))

    for name, res in results_1ycr.items():
        tk = res["topk"]
        row_str = (f"{name:<22} | {res['rank_changed']:<16} | {res['roc_auc']:<8.4f} | {res['auprc']:<8.4f} | "
                   f"{tk[5]['precision']:<6.4f} | {tk[10]['precision']:<6.4f} | {tk[28]['precision']:<6.4f} | {tk[50]['precision']:<6.4f} | "
                   f"{tk[28]['recall']:<6.4f} | {tk[28]['enrichment']:<9.2f}x")
        print(row_str)

    # ==========================================================================
    # 2. EVALUATION ON 14-COMPLEX VALIDATION SET
    # ==========================================================================
    print("\n" + "=" * 90)
    print("                PART 2: 14-COMPLEX HELD-OUT VALIDATION SET AUDIT")
    print("=" * 90)

    # For validation, we evaluate both:
    # A) Pooled Aggregate (concatenating all normalized matrices across the 14 complexes)
    # B) Mean Per-Complex (computing metrics independently for each complex, then averaging)

    results_val_pooled = {}
    results_val_per_complex = {}

    # Total contacts in validation set
    all_val_targets_flat = np.concatenate([c.flatten() for c in val_gt_cmaps])
    total_val_pos = int(np.sum(all_val_targets_flat))
    total_val_pairs = len(all_val_targets_flat)
    gt_val_pos_rate = total_val_pos / total_val_pairs

    topk_val_ratios = [0.001, 0.005, 0.01, 0.02, 0.05]
    topk_val_counts = [int(total_val_pairs * r) for r in topk_val_ratios]

    for name, func in NORMALIZATION_METHODS.items():
        # Apply normalization to each validation complex
        val_norm_mats = [func(m) for m in val_raw_matrices]
        val_norm_flats = [m.flatten() for m in val_norm_mats]
        val_gt_flats = [c.flatten() for c in val_gt_cmaps]

        # --- A) Pooled Evaluation ---
        pooled_probs = np.concatenate(val_norm_flats)
        pooled_targets = np.concatenate(val_gt_flats)

        pooled_auprc = average_precision_score(pooled_targets, pooled_probs)
        pooled_roc_auc = roc_auc_score(pooled_targets, pooled_probs)
        best_tau, best_f1, best_prec, best_rec = find_optimal_threshold(pooled_probs, pooled_targets)
        topk_pooled = calc_topk_metrics(pooled_probs, pooled_targets, topk_val_counts, total_val_pos)

        results_val_pooled[name] = {
            "auprc": pooled_auprc,
            "roc_auc": pooled_roc_auc,
            "best_tau": best_tau,
            "best_f1": best_f1,
            "best_prec": best_prec,
            "best_rec": best_rec,
            "topk": topk_pooled
        }

        # --- B) Per-Complex Evaluation ---
        c_auprcs = []
        c_roc_aucs = []
        c_p10s = []
        c_r10s = []
        c_p20s = []
        c_r20s = []

        for p_flat, t_flat in zip(val_norm_flats, val_gt_flats):
            n_pos = int(np.sum(t_flat))
            if n_pos > 0:
                c_auprc = average_precision_score(t_flat, p_flat)
                c_roc = roc_auc_score(t_flat, p_flat)
            else:
                c_auprc = 0.0
                c_roc = 0.5
            c_auprcs.append(c_auprc)
            c_roc_aucs.append(c_roc)

            # Top-10 & Top-20 per complex
            tk_c = calc_topk_metrics(p_flat, t_flat, [10, 20], n_pos)
            c_p10s.append(tk_c[10]["precision"])
            c_r10s.append(tk_c[10]["recall"])
            c_p20s.append(tk_c[20]["precision"])
            c_r20s.append(tk_c[20]["recall"])

        results_val_per_complex[name] = {
            "mean_auprc": float(np.mean(c_auprcs)),
            "std_auprc": float(np.std(c_auprcs)),
            "mean_roc_auc": float(np.mean(c_roc_aucs)),
            "std_roc_auc": float(np.std(c_roc_aucs)),
            "mean_p10": float(np.mean(c_p10s)),
            "mean_r10": float(np.mean(c_r10s)),
            "mean_p20": float(np.mean(c_p20s)),
            "mean_r20": float(np.mean(c_r20s)),
        }

    # Print Pooled Validation Table
    print("\n--- Pooled Validation Set Performance Across All 14 Complexes ---")
    header_val_pooled = f"{'Strategy':<22} | {'Pooled ROC':<10} | {'Pooled AUPRC':<12} | {'Best Tau':<10} | {'Opt F1':<8} | {'Opt Prec':<8} | {'Opt Rec':<8}"
    print(header_val_pooled)
    print("-" * len(header_val_pooled))

    for name in NORMALIZATION_METHODS.keys():
        vp = results_val_pooled[name]
        print(f"{name:<22} | {vp['roc_auc']:<10.4f} | {vp['auprc']:<12.6f} | {vp['best_tau']:<10.4f} | {vp['best_f1']:<8.4f} | {vp['best_prec']:<8.4f} | {vp['best_rec']:<8.4f}")

    # Print Mean Per-Complex Validation Table
    print("\n--- Mean Per-Complex Validation Performance (Averaged Over 14 Individual Complexes) ---")
    header_val_pc = f"{'Strategy':<22} | {'Mean ROC-AUC':<14} | {'Mean AUPRC':<14} | {'Mean P@10':<10} | {'Mean R@10':<10} | {'Mean P@20':<10} | {'Mean R@20':<10}"
    print(header_val_pc)
    print("-" * len(header_val_pc))

    for name in NORMALIZATION_METHODS.keys():
        vc = results_val_per_complex[name]
        print(f"{name:<22} | {vc['mean_roc_auc']:<14.4f} | {vc['mean_auprc']:<14.6f} | {vc['mean_p10']:<10.4f} | {vc['mean_r10']:<10.4f} | {vc['mean_p20']:<10.4f} | {vc['mean_r20']:<10.4f}")

    # ==========================================================================
    # 3. CROSS-CALIBRATION AUDIT: Evaluate 1YCR using Validation-Tuned Threshold
    # ==========================================================================
    print("\n" + "=" * 90)
    print("     PART 3: 1YCR PERFORMANCE AT VALIDATION-OPTIMIZED DECISION THRESHOLDS")
    print("=" * 90)
    header_cross = f"{'Strategy':<22} | {'Val Opt Tau':<12} | {'1YCR Prec':<10} | {'1YCR Rec':<10} | {'1YCR F1':<10} | {'1YCR Opt F1':<12} | {'F1 Gap':<8}"
    print(header_cross)
    print("-" * len(header_cross))

    for name, func in NORMALIZATION_METHODS.items():
        norm_mat_1ycr = func(raw_matrix_1ycr)
        norm_flat_1ycr = norm_mat_1ycr.flatten()

        val_tau = results_val_pooled[name]["best_tau"]

        preds_1ycr = (norm_flat_1ycr >= val_tau)
        tp = np.sum(preds_1ycr & (gt_1ycr_flat == 1))
        fp = np.sum(preds_1ycr & (gt_1ycr_flat == 0))
        fn = np.sum((~preds_1ycr) & (gt_1ycr_flat == 1))

        prec_val_tau = tp / (tp + fp + 1e-8)
        rec_val_tau = tp / (tp + fn + 1e-8)
        f1_val_tau = 2 * prec_val_tau * rec_val_tau / (prec_val_tau + rec_val_tau + 1e-8) if (prec_val_tau + rec_val_tau) > 0 else 0.0

        opt_f1_1ycr = results_1ycr[name]["best_f1"]
        f1_gap = opt_f1_1ycr - f1_val_tau

        print(f"{name:<22} | {val_tau:<12.4f} | {prec_val_tau:<10.4f} | {rec_val_tau:<10.4f} | {f1_val_tau:<10.4f} | {opt_f1_1ycr:<12.4f} | {f1_gap:<8.4f}")

    # ==========================================================================
    # 4. GENERATE COMPARISON CHARTS
    # ==========================================================================
    assets_dir = PROJECT_ROOT / "assets" / "evaluation"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Precision-Recall Curves on 1YCR for Key Normalization Methods
    plt.figure(figsize=(10, 7), dpi=300)
    styles = {
        "Raw": ("#94A3B8", "--", 1.5),
        "Per-complex Min-Max": ("#2563EB", "-", 2.0),
        "Per-complex Percentile": ("#059669", "-", 2.0),
        "Row-wise": ("#D97706", "-.", 1.8),
        "Column-wise": ("#DC2626", ":", 1.8),
        "Dual (Row x Col)": ("#7C3AED", "-", 2.0),
        "APC (Avg Prod Corr)": ("#DB2777", "-.", 1.8)
    }

    for name, func in NORMALIZATION_METHODS.items():
        norm_mat = func(raw_matrix_1ycr)
        norm_flat = norm_mat.flatten()
        prec_c, rec_c, _ = precision_recall_curve(gt_1ycr_flat, norm_flat)
        auprc_val = results_1ycr[name]["auprc"]
        color, ls, lw = styles.get(name, ("#000000", "-", 1.5))
        plt.plot(rec_c, prec_c, label=f"{name} (AUPRC={auprc_val:.4f})", color=color, linestyle=ls, linewidth=lw)

    gt_pos_rate_1ycr = total_pos_1ycr / len(gt_1ycr_flat)
    plt.axhline(y=gt_pos_rate_1ycr, color="#64748B", linestyle=":", label=f"Random Baseline ({gt_pos_rate_1ycr:.4f})", lw=1.2)

    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("IRLM Normalization Strategy Comparison on PDB 1YCR (TP53-MDM2)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    plot1_path = assets_dir / "irlm_normalization_1ycr_pr_curves.png"
    plt.savefig(plot1_path)
    plt.close()
    print(f"\n1YCR PR Curves saved to: {plot1_path}")

    # Plot 2: Pooled Validation Precision-Recall Curves
    plt.figure(figsize=(10, 7), dpi=300)
    all_val_targets_flat = np.concatenate([c.flatten() for c in val_gt_cmaps])

    for name, func in NORMALIZATION_METHODS.items():
        val_norm_mats = [func(m) for m in val_raw_matrices]
        val_norm_flats = [m.flatten() for m in val_norm_mats]
        pooled_probs = np.concatenate(val_norm_flats)

        prec_c, rec_c, _ = precision_recall_curve(all_val_targets_flat, pooled_probs)
        auprc_val = results_val_pooled[name]["auprc"]
        color, ls, lw = styles.get(name, ("#000000", "-", 1.5))
        plt.plot(rec_c, prec_c, label=f"{name} (AUPRC={auprc_val:.4f})", color=color, linestyle=ls, linewidth=lw)

    plt.axhline(y=gt_val_pos_rate, color="#64748B", linestyle=":", label=f"Random Baseline ({gt_val_pos_rate:.4f})", lw=1.2)

    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("IRLM Normalization Strategy Comparison on 14 Pooled Validation Complexes", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    plot2_path = assets_dir / "irlm_normalization_val_pr_curves.png"
    plt.savefig(plot2_path)
    plt.close()
    print(f"Pooled Validation PR Curves saved to: {plot2_path}")

    print("\nEvaluation pipeline complete successfully!")


if __name__ == "__main__":
    run_eval_pipeline()
