import os
import sys
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.irlm_module import InteractionRegionLocalizationModule


def evaluate_1ycr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npz_path = PROJECT_ROOT / "data" / "processed" / "irlm_dataset" / "1YCR_A_B.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"1YCR dataset artifact not found at {npz_path}")

    model_path = PROJECT_ROOT / "models" / "irlm_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"IRLM checkpoint not found at {model_path}")

    # Load 1YCR NPZ Artifact
    data = np.load(npz_path)
    complex_id = str(data["complex_id"])
    seq_a = str(data["seq_a"])  # MDM2 (85 residues)
    seq_b = str(data["seq_b"])  # TP53 peptide (13 residues, p53 17-29)
    h_a = torch.tensor(data["esm_embedding_a"], dtype=torch.float32).to(device)
    h_b = torch.tensor(data["esm_embedding_b"], dtype=torch.float32).to(device)
    gt_cmap = data["contact_map"]  # (85, 13)
    
    z_a = torch.tensor(data["graph_embedding_a"], dtype=torch.float32).to(device) if "graph_embedding_a" in data else None
    z_b = torch.tensor(data["graph_embedding_b"], dtype=torch.float32).to(device) if "graph_embedding_b" in data else None

    L_A, L_B = gt_cmap.shape
    total_pairs = L_A * L_B
    gt_pos_contacts = int(np.sum(gt_cmap))
    gt_pos_rate = gt_pos_contacts / total_pairs

    print("\n" + "=" * 70)
    print("      IRLM EXTERNAL STRUCTURAL CASE STUDY: TP53 - MDM2 (PDB: 1YCR)")
    print("=" * 70)
    print(f"Complex ID              : {complex_id}")
    print(f"MDM2 (Chain A) Length   : {L_A} aa")
    print(f"TP53 (Chain B) Length   : {L_B} aa (Sequence: {seq_b})")
    print(f"Total Residue Pairs     : {total_pairs}")
    print(f"Ground-Truth Contacts   : {gt_pos_contacts} (C-alpha distance <= 8.0 A)")
    print(f"Ground-Truth Pos Rate   : {gt_pos_rate:.6f} ({gt_pos_rate * 100:.3f}%)")

    # Load Model
    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        r_a, r_b, interaction_matrix_tensor = model.compute_residue_importance(h_a, h_b, z_a, z_b)
        pred_matrix = interaction_matrix_tensor.cpu().numpy()  # (85, 13)
        r_a_np = r_a.cpu().numpy().flatten()
        r_b_np = r_b.cpu().numpy().flatten()

    probs_flat = pred_matrix.flatten()
    targets_flat = gt_cmap.flatten()

    # 1. Global Metrics
    auprc = average_precision_score(targets_flat, probs_flat)
    roc_auc = roc_auc_score(targets_flat, probs_flat)
    random_auprc = gt_pos_rate
    fold_enrichment_auprc = auprc / (random_auprc + 1e-12)

    print("\n--- Model Performance Metrics on 1YCR ---")
    print(f"AUPRC                  : {auprc:.6f} ({fold_enrichment_auprc:.2f}x over random baseline)")
    print(f"ROC-AUC                : {roc_auc:.6f} (Random = 0.500000)")
    print(f"Min Prob Prediction    : {np.min(probs_flat):.6f}")
    print(f"Max Prob Prediction    : {np.max(probs_flat):.6f}")
    print(f"Mean Prob Prediction   : {np.mean(probs_flat):.6f}")
    print(f"Median Prob Prediction : {np.median(probs_flat):.6f}")

    # 2. Performance at Validation-Selected Threshold (tau* = 0.0375)
    val_tau = 0.0375
    preds_val_tau = (probs_flat >= val_tau)
    tp_tau = int(np.sum(preds_val_tau & (targets_flat == 1)))
    fp_tau = int(np.sum(preds_val_tau & (targets_flat == 0)))
    fn_tau = int(np.sum((~preds_val_tau) & (targets_flat == 1)))
    tn_tau = int(np.sum((~preds_val_tau) & (targets_flat == 0)))

    prec_tau = tp_tau / (tp_tau + fp_tau + 1e-8)
    rec_tau = tp_tau / (tp_tau + fn_tau + 1e-8)
    f1_tau = 2 * prec_tau * rec_tau / (prec_tau + rec_tau + 1e-8)

    print(f"\n--- Metrics at Validation Decision Threshold (tau = {val_tau:.4f}) ---")
    print(f"True Positives (TP)    : {tp_tau}")
    print(f"False Positives (FP)   : {fp_tau}")
    print(f"False Negatives (FN)   : {fn_tau}")
    print(f"True Negatives (TN)    : {tn_tau}")
    print(f"Precision              : {prec_tau:.6f}")
    print(f"Recall                 : {rec_tau:.6f}")
    print(f"F1-Score               : {f1_tau:.6f}")

    # 3. Performance at Top-K Predictions
    top_k_list = [5, 10, 20, 28, 50]
    sorted_indices = np.argsort(probs_flat)[::-1]

    print("\n--- Performance at Top-K Pair Predictions ---")
    print(f"{'Top K Pairs':<12} | {'Positives in Top K':<18} | {'Precision@K':<12} | {'Recall@K':<10} | {'F1-Score':<10} | {'Enrichment':<12}")
    print("-" * 82)
    top_k_results = {}
    for k in top_k_list:
        k_idx = sorted_indices[:k]
        pos_in_k = int(np.sum(targets_flat[k_idx] == 1))
        prec_k = pos_in_k / k
        rec_k = pos_in_k / gt_pos_contacts
        f1_k = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-8)
        enrich_k = prec_k / (gt_pos_rate + 1e-12)
        top_k_results[k] = {
            "pos_in_k": pos_in_k, "precision": prec_k, "recall": rec_k, "f1": f1_k, "enrichment": enrich_k
        }
        print(f"K = {k:<8d} | {pos_in_k:<18d} | {prec_k:<12.6f} | {rec_k:<10.6f} | {f1_k:<10.6f} | {enrich_k:<12.2f}x")

    # 4. Identification of Top Predicted Residues
    # TP53 residues: index j in [0..12], amino acid seq_b[j], p53 numbering j + 17
    # Score per TP53 residue: sum or max of interaction probabilities across MDM2
    tp53_scores_sum = np.sum(pred_matrix, axis=0)
    tp53_scores_max = np.max(pred_matrix, axis=0)

    # Count actual ground-truth contacts per TP53 residue
    tp53_gt_contacts = np.sum(gt_cmap, axis=0)

    tp53_residues = []
    for j in range(L_B):
        aa = seq_b[j]
        p53_res_num = j + 17
        tp53_residues.append({
            "idx": j,
            "res_name": f"{aa}{p53_res_num}",
            "aa": aa,
            "p53_num": p53_res_num,
            "importance_r_b": float(r_b_np[j]),
            "sum_prob": float(tp53_scores_sum[j]),
            "max_prob": float(tp53_scores_max[j]),
            "gt_contacts": int(tp53_gt_contacts[j])
        })

    # Sort TP53 residues by model predicted max_prob
    tp53_ranked_by_max = sorted(tp53_residues, key=lambda x: x["max_prob"], reverse=True)
    # Sort TP53 residues by r_b importance
    tp53_ranked_by_rb = sorted(tp53_residues, key=lambda x: x["importance_r_b"], reverse=True)

    print("\n--- TP53 Peptide Residue Predictions (Sorted by Max Prob) ---")
    print(f"{'Rank':<6} | {'Residue':<10} | {'r_b Score':<12} | {'Max Contact Prob':<18} | {'Sum Prob':<12} | {'GT Contacts':<12} | {'Key Triad?':<12}")
    print("-" * 90)
    key_triad = {"F19", "W23", "L26"}
    for rank, res in enumerate(tp53_ranked_by_max, 1):
        is_key = "YES (Key)" if res["res_name"] in key_triad else "No"
        print(f"{rank:<6d} | {res['res_name']:<10} | {res['importance_r_b']:<12.6f} | {res['max_prob']:<18.6f} | {res['sum_prob']:<12.6f} | {res['gt_contacts']:<12d} | {is_key:<12}")

    # MDM2 Residues
    mdm2_scores_sum = np.sum(pred_matrix, axis=1)
    mdm2_scores_max = np.max(pred_matrix, axis=1)
    mdm2_gt_contacts = np.sum(gt_cmap, axis=1)

    mdm2_residues = []
    for i in range(L_A):
        aa = seq_a[i]
        mdm2_res_num = i + 1  # 1YCR chain A numbering 1-85
        mdm2_residues.append({
            "idx": i,
            "res_name": f"{aa}{mdm2_res_num}",
            "aa": aa,
            "mdm2_num": mdm2_res_num,
            "importance_r_a": float(r_a_np[i]),
            "sum_prob": float(mdm2_scores_sum[i]),
            "max_prob": float(mdm2_scores_max[i]),
            "gt_contacts": int(mdm2_gt_contacts[i])
        })

    mdm2_ranked_by_max = sorted(mdm2_residues, key=lambda x: x["max_prob"], reverse=True)

    print("\n--- Top 10 MDM2 Residues (Sorted by Max Prob) ---")
    print(f"{'Rank':<6} | {'Residue':<10} | {'r_a Score':<12} | {'Max Contact Prob':<18} | {'Sum Prob':<12} | {'GT Contacts':<12}")
    print("-" * 80)
    for rank, res in enumerate(mdm2_ranked_by_max[:10], 1):
        print(f"{rank:<6d} | {res['res_name']:<10} | {res['importance_r_a']:<12.6f} | {res['max_prob']:<18.6f} | {res['sum_prob']:<12.6f} | {res['gt_contacts']:<12d}")

    # 5. Check Enrichment of Key Biological Residues (Phe19, Trp23, Leu26)
    print("\n--- Key Biological Residue Audit (Phe19, Trp23, Leu26) ---")
    triad_info = {}
    for res in tp53_residues:
        if res["res_name"] in key_triad:
            # Find rank by max_prob and rank by r_b
            rank_max = [r["res_name"] for r in tp53_ranked_by_max].index(res["res_name"]) + 1
            rank_rb = [r["res_name"] for r in tp53_ranked_by_rb].index(res["res_name"]) + 1
            triad_info[res["res_name"]] = {
                "max_prob": res["max_prob"],
                "rank_max": rank_max,
                "r_b_score": res["importance_r_b"],
                "rank_rb": rank_rb,
                "gt_contacts": res["gt_contacts"]
            }
            print(f"Residue {res['res_name']}: Max Prob = {res['max_prob']:.6f} (Rank {rank_max}/13), r_b = {res['importance_r_b']:.6f} (Rank {rank_rb}/13), GT Contacts = {res['gt_contacts']}")

    # 6. Generate Visualizations
    assets_dir = PROJECT_ROOT / "assets" / "evaluation"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Visualization 1: Predicted Interaction Probability Heatmap
    heatmap_path = assets_dir / "irlm_1ycr_heatmap.png"
    plt.figure(figsize=(10, 16), dpi=300)
    
    tp53_labels = [f"{seq_b[j]}{j+17}" for j in range(L_B)]
    # For MDM2, label every 5th residue to keep it readable
    mdm2_labels = [f"{seq_a[i]}{i+1}" if i % 5 == 0 else "" for i in range(L_A)]

    sns.heatmap(pred_matrix, xticklabels=tp53_labels, yticklabels=mdm2_labels, cmap="YlOrRd", cbar_kws={'label': 'Predicted Contact Probability'})
    plt.title("IRLM Predicted Residue-Residue Interaction Probability Matrix (1YCR: MDM2 vs TP53)", fontsize=13, fontweight="bold")
    plt.xlabel("TP53 Peptide Residues (Chain B, p53: 17-29)", fontsize=11, fontweight="bold")
    plt.ylabel("MDM2 Residues (Chain A, 1-85)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"\nPredicted interaction matrix heatmap saved to: {heatmap_path}")

    # Visualization 2: Side-by-Side Comparison of Predicted vs Experimental Ground-Truth Contact Map
    comparison_path = assets_dir / "irlm_1ycr_contact_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), dpi=300)

    # Panel A: Ground Truth
    sns.heatmap(gt_cmap, ax=axes[0], xticklabels=tp53_labels, yticklabels=mdm2_labels, cmap="Blues", cbar=False)
    axes[0].set_title("A. Experimental Ground-Truth Contact Map\n(C-alpha distance <= 8.0 A, 28 contacts)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("TP53 Residues", fontsize=10)
    axes[0].set_ylabel("MDM2 Residues", fontsize=10)

    # Panel B: IRLM Continuous Predictions
    sns.heatmap(pred_matrix, ax=axes[1], xticklabels=tp53_labels, yticklabels=mdm2_labels, cmap="YlOrRd", cbar=True, cbar_kws={'label': 'Probability'})
    axes[1].set_title("B. IRLM Model Predictions\n(Residue-Residue Contact Probabilities)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("TP53 Residues", fontsize=10)
    axes[1].set_ylabel("")

    # Panel C: Binary Predictions at tau = 0.0375 overlaid with Ground Truth
    # 0 = TN, 1 = FP (red), 2 = FN (blue), 3 = TP (green)
    overlay = np.zeros_like(gt_cmap, dtype=int)
    pred_bin = (pred_matrix >= val_tau)
    overlay[(pred_bin == False) & (gt_cmap == 0)] = 0  # TN
    overlay[(pred_bin == True) & (gt_cmap == 0)] = 1   # FP
    overlay[(pred_bin == False) & (gt_cmap == 1)] = 2  # FN
    overlay[(pred_bin == True) & (gt_cmap == 1)] = 3   # TP

    from matplotlib.colors import ListedColormap
    # Colors: 0: white, 1: light coral (FP), 2: light sky blue (FN), 3: dark green (TP)
    cmap_custom = ListedColormap(["#F8FAFC", "#FCA5A5", "#93C5FD", "#15803D"])
    sns.heatmap(overlay, ax=axes[2], xticklabels=tp53_labels, yticklabels=mdm2_labels, cmap=cmap_custom, cbar=False)
    axes[2].set_title(f"C. Prediction vs Ground Truth at tau={val_tau:.4f}\n(Green=TP, Coral=FP, Blue=FN, White=TN)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("TP53 Residues", fontsize=10)
    axes[2].set_ylabel("")

    plt.suptitle("PDB 1YCR (MDM2 - TP53) IRLM Structural Evaluation Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    print(f"Side-by-side contact comparison saved to: {comparison_path}")

    # Summary Conclusion
    print("\n" + "=" * 70)
    print("                     CASE STUDY CONCLUSION")
    print("=" * 70)
    print(f"ROC-AUC: {roc_auc:.4f} | AUPRC: {auprc:.4f} ({fold_enrichment_auprc:.2f}x over random)")
    print(f"Top 28 Pairs Precision: {top_k_results[28]['precision']:.4f} ({top_k_results[28]['enrichment']:.2f}x random enrichment)")
    print("Key Triad Recovery Audit:")
    for k_res, k_val in triad_info.items():
        print(f"  - {k_res}: Rank {k_val['rank_max']}/13 by max prob (Prob = {k_val['max_prob']:.4f}), GT contacts = {k_val['gt_contacts']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_1ycr()
