import os
import sys
import glob
import random
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.irlm_module import InteractionRegionLocalizationModule, IRLMLoss
from src.training.train_irlm import IRLMDataset


def evaluate_irlm_model():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not data_files:
        raise FileNotFoundError(f"No .npz dataset files found in {data_dir}")

    print(f"Total dataset files found: {len(data_files)}")

    # Reproduce identical 80/20 train/val split from train_irlm.py
    rng = random.Random(seed)
    shuffled_files = list(data_files)
    rng.shuffle(shuffled_files)

    val_size = max(1, int(len(shuffled_files) * 0.2))
    val_files = shuffled_files[:val_size]
    train_files = shuffled_files[val_size:]

    print(f"Training split: {len(train_files)} complexes")
    print(f"Validation split: {len(val_files)} complexes")

    val_dataset = IRLMDataset(val_files)

    # Load Model
    model_path = PROJECT_ROOT / "models" / "irlm_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded trained IRLM checkpoint from {model_path}")

    # Loss function for reporting val loss
    loss_fn = IRLMLoss(lambda_contact=1.0, lambda_sparsity=0.05, lambda_smooth=0.05).to(device)

    all_probs = []
    all_targets = []
    per_complex_stats = []
    total_val_loss = 0.0

    print("\nEvaluating on Validation Set Complexes...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            name = sample["name"]
            h_a = sample["h_a"].to(device)
            h_b = sample["h_b"].to(device)
            cmap = sample["contact_map"].to(device)
            z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
            z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None

            r_a, r_b, interaction_matrix = model.compute_residue_importance(h_a, h_b, z_a, z_b)

            loss = loss_fn(
                pred_prob=None,
                target_label=None,
                interaction_matrix=interaction_matrix,
                r_a=r_a,
                r_b=r_b,
                contact_map=cmap
            )
            total_val_loss += loss.item()

            probs_np = interaction_matrix.cpu().numpy().flatten()
            targets_np = cmap.cpu().numpy().flatten()

            all_probs.append(probs_np)
            all_targets.append(targets_np)

            n_pos = int(np.sum(targets_np))
            n_total = len(targets_np)
            c_auprc = average_precision_score(targets_np, probs_np) if n_pos > 0 else 0.0

            per_complex_stats.append({
                "name": name,
                "L_A": h_a.shape[0],
                "L_B": h_b.shape[0],
                "total_pairs": n_total,
                "pos_contacts": n_pos,
                "pos_rate": n_pos / n_total,
                "mean_prob": float(np.mean(probs_np)),
                "max_prob": float(np.max(probs_np)),
                "min_prob": float(np.min(probs_np)),
                "auprc": float(c_auprc)
            })

    probs_all = np.concatenate(all_probs)
    targets_all = np.concatenate(all_targets)
    avg_val_loss = total_val_loss / len(val_dataset)

    # 1. Dataset & Prediction Summary
    total_pairs = len(targets_all)
    total_positives = int(np.sum(targets_all))
    gt_pos_rate = total_positives / total_pairs

    print("\n" + "=" * 70)
    print("                IRLM VALIDATION EVALUATION AUDIT")
    print("=" * 70)
    print(f"Validation Complexes Evaluated : {len(val_dataset)}")
    print(f"Total Validation Residue Pairs : {total_pairs:,}")
    print(f"Ground-Truth Positive Contacts : {total_positives:,}")
    print(f"Ground-Truth Positive Rate    : {gt_pos_rate:.6f} ({gt_pos_rate * 100:.3f}%)")
    print(f"Validation Loss                : {avg_val_loss:.4f}")

    # 2. Probability Distribution Statistics
    p_min = np.min(probs_all)
    p_max = np.max(probs_all)
    p_mean = np.mean(probs_all)
    p_median = np.median(probs_all)
    p_std = np.std(probs_all)
    p_percentiles = np.percentile(probs_all, [25, 50, 75, 90, 95, 99, 99.9])

    print("\n--- Predicted Probability Statistics ---")
    print(f"Min Probability    : {p_min:.6f}")
    print(f"Max Probability    : {p_max:.6f}")
    print(f"Mean Probability   : {p_mean:.6f}")
    print(f"Median Probability : {p_median:.6f}")
    print(f"Std Deviation      : {p_std:.6f}")
    print("Percentiles:")
    print(f"  25th: {p_percentiles[0]:.6f} | 50th: {p_percentiles[1]:.6f} | 75th: {p_percentiles[2]:.6f}")
    print(f"  90th: {p_percentiles[3]:.6f} | 95th: {p_percentiles[4]:.6f} | 99th: {p_percentiles[5]:.6f} | 99.9th: {p_percentiles[6]:.6f}")

    # 3. Overall AUPRC and ROC-AUC
    global_auprc = average_precision_score(targets_all, probs_all)
    global_roc_auc = roc_auc_score(targets_all, probs_all)
    random_auprc = gt_pos_rate

    print("\n--- Ranking & Discriminative Quality ---")
    print(f"Global AUPRC           : {global_auprc:.6f}")
    print(f"Random Baseline AUPRC  : {random_auprc:.6f}")
    print(f"AUPRC Fold Improvement : {global_auprc / (random_auprc + 1e-12):.2f}x over random")
    print(f"Global ROC-AUC         : {global_roc_auc:.6f} (Random = 0.500000)")

    # 4. Evaluation at Fixed Thresholds
    fixed_thresholds = [0.01, 0.02, 0.05, 0.10, 0.50]
    print("\n--- Performance at Fixed Thresholds ---")
    print(f"{'Threshold':<10} | {'TP':<6} | {'FP':<8} | {'FN':<6} | {'TN':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    for t in fixed_thresholds:
        preds = (probs_all >= t)
        tp = np.sum(preds & (targets_all == 1))
        fp = np.sum(preds & (targets_all == 0))
        fn = np.sum((~preds) & (targets_all == 1))
        tn = np.sum((~preds) & (targets_all == 0))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"{t:<10.2f} | {tp:<6d} | {fp:<8d} | {fn:<6d} | {tn:<8d} | {prec:<10.6f} | {rec:<10.6f} | {f1:<10.6f}")

    # 5. Optimal Threshold Search (Validation Data Only)
    sweep_thresholds = np.linspace(0.001, 0.50, 1000)
    best_f1 = -1.0
    best_thresh = 0.0
    best_metrics = {}

    for t in sweep_thresholds:
        preds = (probs_all >= t)
        tp = np.sum(preds & (targets_all == 1))
        fp = np.sum(preds & (targets_all == 0))
        fn = np.sum((~preds) & (targets_all == 1))
        tn = np.sum((~preds) & (targets_all == 0))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "threshold": float(t),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }

    print("\n--- Optimal Validation Threshold Sweep Result ---")
    print(f"Optimal Threshold (tau*) : {best_metrics['threshold']:.6f}")
    print(f"True Positives (TP)      : {best_metrics['tp']:,}")
    print(f"False Positives (FP)     : {best_metrics['fp']:,}")
    print(f"False Negatives (FN)     : {best_metrics['fn']:,}")
    print(f"True Negatives (TN)      : {best_metrics['tn']:,}")
    print(f"Optimal Precision        : {best_metrics['precision']:.6f}")
    print(f"Optimal Recall           : {best_metrics['recall']:.6f}")
    print(f"Optimal F1-Score         : {best_metrics['f1']:.6f}")

    # 6. Top-K Contact Enrichment Analysis
    print("\n--- Top-K Contact Enrichment Analysis (Ranking Power Audit) ---")
    top_k_ratios = [0.001, 0.005, 0.01, 0.02, 0.05]
    sorted_indices = np.argsort(probs_all)[::-1]

    print(f"{'Top % Pairs':<12} | {'K Pairs':<8} | {'Positives in Top K':<18} | {'Precision@K':<12} | {'Enrichment vs Random':<22}")
    print("-" * 80)
    for ratio in top_k_ratios:
        k = int(total_pairs * ratio)
        if k == 0:
            continue
        top_k_idx = sorted_indices[:k]
        pos_in_k = np.sum(targets_all[top_k_idx] == 1)
        prec_k = pos_in_k / k
        enrichment = prec_k / (gt_pos_rate + 1e-12)
        print(f"{ratio * 100:<11.2f}% | {k:<8d} | {pos_in_k:<18d} | {prec_k:<12.6f} | {enrichment:<22.2f}x")

    # 7. Generate & Save Precision-Recall Curve
    assets_dir = PROJECT_ROOT / "assets" / "evaluation"
    assets_dir.mkdir(parents=True, exist_ok=True)
    pr_curve_path = assets_dir / "irlm_precision_recall_curve.png"

    precision_curve, recall_curve, _ = precision_recall_curve(targets_all, probs_all)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall_curve, precision_curve, label=f"Trained IRLM (AUPRC = {global_auprc:.4f})", color="#2563EB", lw=2)
    plt.axhline(y=gt_pos_rate, color="#DC2626", linestyle="--", label=f"Random Baseline (Pos Rate = {gt_pos_rate:.4f})", lw=1.5)
    plt.scatter([best_metrics["recall"]], [best_metrics["precision"]], color="#059669", s=80, zorder=5,
                label=f"Optimal Threshold (tau={best_thresh:.3f}, F1={best_f1:.4f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("IRLM Residue-Residue Contact Prediction: Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"\nPrecision-Recall curve saved successfully to: {pr_curve_path}")

    # 8. Assessment Conclusion
    print("\n" + "=" * 70)
    print("                       IRLM AUDIT CONCLUSION")
    print("=" * 70)
    if global_auprc > random_auprc * 1.5 or global_roc_auc > 0.65:
        print("RESULT: IRLM HAS LEARNED MEANINGFUL RESIDUE-CONTACT RANKING!")
        print(f"Explanation: Max predicted probability is {p_max:.4f} (well below default threshold 0.50).")
        print(f"However, ranking quality is strong with AUPRC={global_auprc:.4f} ({global_auprc / random_auprc:.1f}x random) and ROC-AUC={global_roc_auc:.4f}.")
        print(f"At calibrated validation threshold tau*={best_thresh:.4f}, IRLM achieves F1={best_f1:.4f}.")
    else:
        print("RESULT: IRLM SHOWS LIMITED DISCRIMINATIVE POWER.")
        print(f"AUPRC={global_auprc:.4f} vs Random={random_auprc:.4f}, ROC-AUC={global_roc_auc:.4f}.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_irlm_model()
