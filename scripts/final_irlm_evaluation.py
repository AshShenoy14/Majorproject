import os
import sys
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.irlm_module import InteractionRegionLocalizationModule
from src.training.train_irlm import IRLMDataset

def run_final_irlm_evaluation():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing IRLM Final Evaluation on device: {device}")

    data_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    data_files = sorted(glob.glob(str(data_dir / "*.npz")))
    if not data_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    # Reproduce exact 80/20 train/val split (seed=42)
    rng = random.Random(seed)
    shuffled_files = list(data_files)
    rng.shuffle(shuffled_files)

    val_size = max(1, int(len(shuffled_files) * 0.2))
    val_files = shuffled_files[:val_size]
    print(f"Validation dataset split: {len(val_files)} complexes")

    val_dataset = IRLMDataset(val_files)

    # Load trained IRLM checkpoint
    model_path = PROJECT_ROOT / "models" / "irlm_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"IRLM model checkpoint not found at {model_path}")

    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs_per_complex = []
    targets_per_complex = []
    complex_names = []

    with torch.no_grad():
        for sample in val_dataset:
            name = sample["name"]
            h_a = sample["h_a"].to(device)
            h_b = sample["h_b"].to(device)
            cmap = sample["contact_map"].numpy()

            z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
            z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None

            r_a, r_b, interaction_matrix = model.compute_residue_importance(h_a, h_b, z_a, z_b)
            p_mat = interaction_matrix.cpu().numpy()

            probs_per_complex.append(p_mat)
            targets_per_complex.append(cmap)
            complex_names.append(name)

    all_p = np.concatenate([p.flatten() for p in probs_per_complex])
    all_t = np.concatenate([t.flatten() for t in targets_per_complex])

    total_val_pairs = len(all_p)
    total_val_contacts = int(np.sum(all_t))
    bg_rate = total_val_contacts / total_val_pairs if total_val_pairs > 0 else 0.0

    auroc = float(roc_auc_score(all_t, all_p))
    auprc = float(average_precision_score(all_t, all_p))

    print(f"Validation Pairs: {total_val_pairs:,} | Val Contacts: {total_val_contacts:,} ({bg_rate*100:.4f}%)")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    # Threshold Sweeping: 0.001 to 0.99 (combining fine low-prob steps and full range)
    low_ths = np.linspace(0.001, 0.10, 100)
    high_ths = np.linspace(0.11, 0.99, 89)
    thresholds_to_sweep = np.sort(np.unique(np.concatenate([low_ths, high_ths])))

    sweep_results = []
    best_f1 = -1.0
    best_th_metrics = {}
    metrics_05 = {}

    for th in thresholds_to_sweep:
        th_val = float(th)
        preds = (all_p >= th_val).astype(int)
        tp = int(np.sum((preds == 1) & (all_t == 1)))
        fp = int(np.sum((preds == 1) & (all_t == 0)))
        fn = int(np.sum((preds == 0) & (all_t == 1)))
        tn = int(np.sum((preds == 0) & (all_t == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        sparsity = float(tn + fn) / total_val_pairs if total_val_pairs > 0 else 1.0
        recovery_rate = float(tp / total_val_contacts) if total_val_contacts > 0 else 0.0

        item = {
            "threshold": th_val,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "predicted_sparsity": sparsity,
            "contact_recovery_rate": recovery_rate,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }
        sweep_results.append(item)

        if f1 > best_f1:
            best_f1 = f1
            best_th_metrics = item

        if abs(th_val - 0.50) < 0.005 and not metrics_05:
            metrics_05 = item

    if not metrics_05:
        # Default fallback for 0.50 exact
        preds = (all_p >= 0.50).astype(int)
        tp = int(np.sum((preds == 1) & (all_t == 1)))
        fp = int(np.sum((preds == 1) & (all_t == 0)))
        fn = int(np.sum((preds == 0) & (all_t == 1)))
        tn = int(np.sum((preds == 0) & (all_t == 0)))
        metrics_05 = {
            "threshold": 0.50,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1_score": float(2 * tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0.0,
            "predicted_sparsity": float(tn + fn) / total_val_pairs,
            "contact_recovery_rate": float(tp / total_val_contacts) if total_val_contacts > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

    optimal_threshold = best_th_metrics["threshold"]

    # Top-K Evaluation
    top_k_values = [5, 10, 20, 50]
    top_k_metrics = {}

    for k in top_k_values:
        precs_k = []
        recs_k = []
        efs_k = []
        for p_mat, t_mat in zip(probs_per_complex, targets_per_complex):
            flat_p = p_mat.flatten()
            flat_t = t_mat.flatten()
            n_pos = flat_t.sum()
            bg_dens = n_pos / len(flat_t) if len(flat_t) > 0 else 0.0

            top_indices = np.argsort(flat_p)[::-1][:k]
            top_t = flat_t[top_indices]

            pk = float(top_t.sum() / k)
            rk = float(top_t.sum() / n_pos) if n_pos > 0 else 0.0
            efk = float(pk / bg_dens) if bg_dens > 0 else 0.0

            precs_k.append(pk)
            recs_k.append(rk)
            efs_k.append(efk)

        top_k_metrics[f"top_{k}"] = {
            "k": k,
            "precision_at_k": float(np.mean(precs_k)),
            "recall_at_k": float(np.mean(recs_k)),
            "enrichment_factor": float(np.mean(efs_k))
        }

    # Prepare Asset Directory
    out_dir = PROJECT_ROOT / "assets" / "evaluation" / "irlm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot: Threshold Sweeps
    plt.figure(figsize=(9, 5), dpi=300)
    ths = [x["threshold"] for x in sweep_results]
    f1s = [x["f1_score"] for x in sweep_results]
    precs = [x["precision"] for x in sweep_results]
    recs = [x["recall"] for x in sweep_results]

    plt.plot(ths, f1s, label="F1 Score", color="#3b82f6", linewidth=2.5)
    plt.plot(ths, precs, label="Precision", color="#10b981", linewidth=2)
    plt.plot(ths, recs, label="Recall", color="#f59e0b", linewidth=2)
    plt.axvline(x=optimal_threshold, color="#ef4444", linestyle="--", label=f"Optimal Thresh ({optimal_threshold:.4f})")
    plt.axvline(x=0.50, color="#6b7280", linestyle=":", label="Default Thresh (0.50)")

    plt.title("IRLM Threshold Sweeping (Validation Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Interaction Score Threshold", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.xlim([0.0, 0.3])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweeps.png")
    plt.close()

    # 2. Plot: Precision-Recall Curve
    prec_curve, rec_curve, _ = precision_recall_curve(all_t, all_p)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(rec_curve, prec_curve, color="#8b5cf6", linewidth=2.5, label=f"IRLM (AUPRC = {auprc:.4f})")
    plt.axhline(y=bg_rate, color="#9ca3af", linestyle="--", label=f"Random Baseline ({bg_rate*100:.3f}%)")
    plt.title("IRLM Precision-Recall Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png")
    plt.close()

    # 3. Plot: ROC Curve
    fpr_curve, tpr_curve, _ = roc_curve(all_t, all_p)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr_curve, tpr_curve, color="#2563eb", linewidth=2.5, label=f"IRLM (AUROC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", label="Random Chance (0.50)")
    plt.title("IRLM Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    # 4. Plot: Top-K Enrichment
    plt.figure(figsize=(8, 5), dpi=300)
    ks = [str(k) for k in top_k_values]
    ef_vals = [top_k_metrics[f"top_{k}"]["enrichment_factor"] for k in top_k_values]
    p_vals = [top_k_metrics[f"top_{k}"]["precision_at_k"] * 100 for k in top_k_values]

    x_indices = np.arange(len(ks))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax1.bar(x_indices - width/2, ef_vals, width, label="Enrichment Factor (x)", color="#6366f1")
    ax1.set_ylabel("Enrichment Factor over Random", color="#4338ca", fontsize=11, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor="#4338ca")
    ax1.set_xlabel("Top K Contact Candidates per Complex", fontsize=11)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x_indices + width/2, p_vals, width, label="Precision @ K (%)", color="#10b981")
    ax2.set_ylabel("Precision @ K (%)", color="#047857", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor="#047857")

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"Top-{k}" for k in ks])
    plt.title("IRLM Top-K Contact Recovery & Enrichment Factor", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_dir / "top_k_enrichment.png")
    plt.close()

    # Save metrics summary JSON
    metrics_summary = {
        "total_val_complexes": len(val_files),
        "total_val_pairs": total_val_pairs,
        "total_val_contacts": total_val_contacts,
        "random_background_density": bg_rate,
        "auroc": auroc,
        "auprc": auprc,
        "optimal_threshold": optimal_threshold,
        "metrics_at_optimal_threshold": best_th_metrics,
        "metrics_at_default_0_5_threshold": metrics_05,
        "top_k_metrics": top_k_metrics,
        "threshold_sweep_data": sweep_results
    }

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n==========================================================================")
    print("                    IRLM FINAL EVALUATION SUMMARY                         ")
    print("==========================================================================")
    print(f" Validation Complexes          : {len(val_files)}")
    print(f" Total Residue Pairs           : {total_val_pairs:,}")
    print(f" Ground Truth Contacts         : {total_val_contacts:,} ({bg_rate*100:.4f}%)")
    print(f" Area Under ROC (AUROC)        : {auroc:.4f}")
    print(f" Area Under PR (AUPRC)         : {auprc:.4f}")
    print("--------------------------------------------------------------------------")
    print(f" Optimal Threshold (F1 Max)    : {optimal_threshold:.4f}")
    print(f"   Precision @ Optimal         : {best_th_metrics['precision']*100:.2f}%")
    print(f"   Recall @ Optimal            : {best_th_metrics['recall']*100:.2f}%")
    print(f"   F1 Score @ Optimal          : {best_th_metrics['f1_score']:.4f}")
    print(f"   Contact Recovery Rate       : {best_th_metrics['contact_recovery_rate']*100:.2f}%")
    print("--------------------------------------------------------------------------")
    print(" Default Threshold (0.50):")
    print(f"   Precision @ 0.50            : {metrics_05['precision']*100:.2f}%")
    print(f"   Recall @ 0.50               : {metrics_05['recall']*100:.2f}%")
    print(f"   Predicted Sparsity @ 0.50   : {metrics_05['predicted_sparsity']*100:.4f}%")
    print("--------------------------------------------------------------------------")
    print(" Top-K Enrichment Factors:")
    for k in top_k_values:
        m = top_k_metrics[f"top_{k}"]
        print(f"   Top-{k:2d}: Prec={m['precision_at_k']*100:.2f}%, Rec={m['recall_at_k']*100:.2f}%, Enrichment={m['enrichment_factor']:.2f}x")
    print("--------------------------------------------------------------------------")
    print(f" Saved Assets in               : {out_dir}")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_final_irlm_evaluation()
