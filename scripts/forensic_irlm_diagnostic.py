import os
import sys
import glob
import random
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.irlm_module import InteractionRegionLocalizationModule, IRLMLoss
from src.training.train_irlm import IRLMDataset

def run_forensic_diagnostic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FORENSIC] Using device: {device}")

    data_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not data_files:
        raise FileNotFoundError(f"No dataset files in {data_dir}")

    # 1. Split integrity audit
    rng = random.Random(seed)
    shuffled_files = list(data_files)
    rng.shuffle(shuffled_files)

    val_size = max(1, int(len(shuffled_files) * 0.2))
    val_files = shuffled_files[:val_size]
    train_files = shuffled_files[val_size:]

    ycr_in_val = any("1YCR" in f for f in val_files)
    ycr_in_train = any("1YCR" in f for f in train_files)

    print("\n" + "=" * 75)
    print(" 1. DATASET SPLIT INTEGRITY AUDIT")
    print("=" * 75)
    print(f"Total complexes in dataset : {len(data_files)}")
    print(f"Training split count       : {len(train_files)}")
    print(f"Validation split count     : {len(val_files)}")
    print(f"1YCR present in Train set? : {ycr_in_train}")
    print(f"1YCR present in Val set?   : {ycr_in_val}")
    assert ycr_in_val and not ycr_in_train, "Split leak detection failed! 1YCR is not cleanly in validation set."
    print("STATUS: PASSED (1YCR is strictly in validation set, zero training leakage)")

    # Load Model
    model_path = PROJECT_ROOT / "models" / "irlm_best.pth"
    model = InteractionRegionLocalizationModule(embed_dim=480).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_dataset = IRLMDataset(val_files)

    # 2. Ground-truth contact density & length comparison
    print("\n" + "=" * 75)
    print(" 2. GROUND-TRUTH CONTACT DENSITY & SEQUENCE LENGTH AUDIT")
    print("=" * 75)

    val_densities = []
    val_total_pairs = []
    val_pos_contacts = []
    val_lengths_a = []
    val_lengths_b = []
    val_length_ratios = []

    per_val_complex_stats = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            name = sample["name"]
            h_a = sample["h_a"].to(device)
            h_b = sample["h_b"].to(device)
            cmap = sample["contact_map"].cpu().numpy()
            
            L_A, L_B = cmap.shape
            n_pairs = L_A * L_B
            n_pos = int(np.sum(cmap))
            pos_rate = n_pos / n_pairs if n_pairs > 0 else 0.0

            r_a, r_b, matrix_tensor = model.compute_residue_importance(
                h_a, h_b,
                sample["z_a"].to(device) if sample["z_a"] is not None else None,
                sample["z_b"].to(device) if sample["z_b"] is not None else None
            )
            probs = matrix_tensor.cpu().numpy().flatten()

            val_densities.append(pos_rate)
            val_total_pairs.append(n_pairs)
            val_pos_contacts.append(n_pos)
            val_lengths_a.append(L_A)
            val_lengths_b.append(L_B)
            val_length_ratios.append(max(L_A, L_B) / max(1, min(L_A, L_B)))

            per_val_complex_stats.append({
                "name": name,
                "L_A": L_A,
                "L_B": L_B,
                "pairs": n_pairs,
                "pos": n_pos,
                "pos_rate": pos_rate,
                "mean_prob": float(np.mean(probs)),
                "max_prob": float(np.max(probs)),
                "min_prob": float(np.min(probs)),
                "std_prob": float(np.std(probs)),
                "p99_prob": float(np.percentile(probs, 99)),
                "esm_norm_a": float(torch.norm(h_a, dim=-1).mean().item()),
                "esm_norm_b": float(torch.norm(h_b, dim=-1).mean().item())
            })

    # Load 1YCR specifically
    ycr_path = PROJECT_ROOT / "data" / "processed" / "irlm_dataset" / "1YCR_A_B.npz"
    ycr_data = np.load(ycr_path)
    ycr_h_a = torch.tensor(ycr_data["esm_embedding_a"], dtype=torch.float32).to(device)
    ycr_h_b = torch.tensor(ycr_data["esm_embedding_b"], dtype=torch.float32).to(device)
    ycr_cmap = ycr_data["contact_map"]
    ycr_z_a = torch.tensor(ycr_data["graph_embedding_a"], dtype=torch.float32).to(device) if "graph_embedding_a" in ycr_data else None
    ycr_z_b = torch.tensor(ycr_data["graph_embedding_b"], dtype=torch.float32).to(device) if "graph_embedding_b" in ycr_data else None

    ycr_LA, ycr_LB = ycr_cmap.shape
    ycr_pairs = ycr_LA * ycr_LB
    ycr_pos = int(np.sum(ycr_cmap))
    ycr_pos_rate = ycr_pos / ycr_pairs

    with torch.no_grad():
        _, _, ycr_matrix = model.compute_residue_importance(ycr_h_a, ycr_h_b, ycr_z_a, ycr_z_b)
        ycr_probs = ycr_matrix.cpu().numpy().flatten()

    print(f"Validation Complexes Count: {len(val_dataset)}")
    print(f"{'Complex':<15} | {'L_A':<5} | {'L_B':<5} | {'Pairs':<8} | {'Pos':<5} | {'Pos Rate':<10} | {'Max Prob':<10} | {'P99 Prob':<10}")
    print("-" * 80)
    for c in per_val_complex_stats:
        print(f"{c['name']:<15} | {c['L_A']:<5d} | {c['L_B']:<5d} | {c['pairs']:<8d} | {c['pos']:<5d} | {c['pos_rate']:<10.6f} | {c['max_prob']:<10.6f} | {c['p99_prob']:<10.6f}")

    print("-" * 80)
    print(f"{'1YCR (MDM2-p53)':<15} | {ycr_LA:<5d} | {ycr_LB:<5d} | {ycr_pairs:<8d} | {ycr_pos:<5d} | {ycr_pos_rate:<10.6f} | {np.max(ycr_probs):<10.6f} | {np.percentile(ycr_probs, 99):<10.6f}")

    print("\nSummary Density & Geometry Metrics:")
    print(f"Validation Mean Pos Rate    : {np.mean(val_densities):.6f} (Min: {np.min(val_densities):.6f}, Max: {np.max(val_densities):.6f})")
    print(f"1YCR Pos Rate              : {ycr_pos_rate:.6f} ({ycr_pos_rate / np.mean(val_densities):.2f}x average validation density!)")
    print(f"Validation Mean Pairs       : {np.mean(val_total_pairs):.1f} (Min: {np.min(val_total_pairs)}, Max: {np.max(val_total_pairs)})")
    print(f"1YCR Total Pairs            : {ycr_pairs} (Much smaller search space: {ycr_pairs / np.mean(val_total_pairs):.2f}x of mean val complex)")
    print(f"Validation Mean L_A/L_B Ratio: {np.mean(val_length_ratios):.2f} (Min: {np.min(val_length_ratios):.2f}, Max: {np.max(val_length_ratios):.2f})")
    print(f"1YCR L_A/L_B Length Ratio   : {ycr_LA / ycr_LB:.2f} (85 aa vs 13 aa peptide: severe length asymmetry!)")

    # 3. Probability distribution audit
    print("\n" + "=" * 75)
    print(" 3. PREDICTED PROBABILITY DISTRIBUTION AUDIT")
    print("=" * 75)

    val_probs_list = []
    val_targets_list = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            h_a = sample["h_a"].to(device)
            h_b = sample["h_b"].to(device)
            z_a = sample["z_a"].to(device) if sample["z_a"] is not None else None
            z_b = sample["z_b"].to(device) if sample["z_b"] is not None else None
            _, _, m = model.compute_residue_importance(h_a, h_b, z_a, z_b)
            val_probs_list.append(m.cpu().numpy().flatten())
            val_targets_list.append(sample["contact_map"].cpu().numpy().flatten())

    val_probs_all = np.concatenate(val_probs_list)
    val_targets_all = np.concatenate(val_targets_list)

    print(f"Metric                       | Validation Set (14 complexes) | 1YCR (TP53-MDM2)")
    print("-" * 75)
    print(f"Total Residue Pairs          | {len(val_probs_all):<29,d} | {len(ycr_probs):<15,d}")
    print(f"Min Probability              | {np.min(val_probs_all):<29.6f} | {np.min(ycr_probs):<15.6f}")
    print(f"Max Probability              | {np.max(val_probs_all):<29.6f} | {np.max(ycr_probs):<15.6f}")
    print(f"Mean Probability             | {np.mean(val_probs_all):<29.6f} | {np.mean(ycr_probs):<15.6f}")
    print(f"Median Probability           | {np.median(val_probs_all):<29.6f} | {np.median(ycr_probs):<15.6f}")
    print(f"Std Deviation                | {np.std(val_probs_all):<29.6f} | {np.std(ycr_probs):<15.6f}")
    print(f"90th Percentile              | {np.percentile(val_probs_all, 90):<29.6f} | {np.percentile(ycr_probs, 90):<15.6f}")
    print(f"95th Percentile              | {np.percentile(val_probs_all, 95):<29.6f} | {np.percentile(ycr_probs, 95):<15.6f}")
    print(f"99th Percentile              | {np.percentile(val_probs_all, 99):<29.6f} | {np.percentile(ycr_probs, 99):<15.6f}")
    print(f"99.9th Percentile            | {np.percentile(val_probs_all, 99.9):<29.6f} | {np.percentile(ycr_probs, 99.9):<15.6f}")

    val_pos_mask = (val_targets_all == 1.0)
    val_neg_mask = (val_targets_all == 0.0)
    ycr_pos_mask = (ycr_cmap.flatten() == 1.0)
    ycr_neg_mask = (ycr_cmap.flatten() == 0.0)

    print("\nMean Probabilities by Ground-Truth Label:")
    print(f"Val True Positive Contacts   : Mean Prob = {np.mean(val_probs_all[val_pos_mask]):.6f}")
    print(f"Val True Negative Pairs      : Mean Prob = {np.mean(val_probs_all[val_neg_mask]):.6f}")
    print(f"Ratio Pos/Neg Mean (Val)     : {np.mean(val_probs_all[val_pos_mask]) / np.mean(val_probs_all[val_neg_mask]):.2f}x")
    print(f"1YCR True Positive Contacts  : Mean Prob = {np.mean(ycr_probs[ycr_pos_mask]):.6f}")
    print(f"1YCR True Negative Pairs     : Mean Prob = {np.mean(ycr_probs[ycr_neg_mask]):.6f}")
    print(f"Ratio Pos/Neg Mean (1YCR)    : {np.mean(ycr_probs[ycr_pos_mask]) / np.mean(ycr_probs[ycr_neg_mask]):.2f}x")

    # 4. Chain orientation & order swap audit
    print("\n" + "=" * 75)
    print(" 4. CHAIN ORIENTATION & ORDER SWAP AUDIT")
    print("=" * 75)

    # In 1YCR: h_a is MDM2 (85 aa), h_b is TP53 (13 aa)
    with torch.no_grad():
        r_a_fwd, r_b_fwd, matrix_fwd = model.compute_residue_importance(ycr_h_a, ycr_h_b, ycr_z_a, ycr_z_b)
        # Swapped: h_a=TP53 (13 aa), h_b=MDM2 (85 aa)
        r_a_swp, r_b_swp, matrix_swp = model.compute_residue_importance(ycr_h_b, ycr_h_a, ycr_z_b, ycr_z_a)

    print(f"Forward Pass (MDM2=A, TP53=B): Matrix shape = {list(matrix_fwd.shape)}")
    print(f"Swapped Pass (TP53=A, MDM2=B): Matrix shape = {list(matrix_swp.shape)} (Transposed shape = {list(matrix_swp.T.shape)})")

    matrix_fwd_np = matrix_fwd.cpu().numpy()
    matrix_swp_transposed_np = matrix_swp.T.cpu().numpy()

    abs_diff = np.abs(matrix_fwd_np - matrix_swp_transposed_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    corr = np.corrcoef(matrix_fwd_np.flatten(), matrix_swp_transposed_np.flatten())[0, 1]

    print(f"Max Absolute Difference (Fwd vs Swapped.T)  : {max_diff:.8f}")
    print(f"Mean Absolute Difference (Fwd vs Swapped.T) : {mean_diff:.8f}")
    print(f"Pearson Correlation (Fwd vs Swapped.T)     : {corr:.8f}")
    print(f"Chain Order Invariance Status               : {'PERFECT (Symmetric)' if corr > 0.99 else 'ASYMMETRIC'}")

    # 5. Feature embedding audit
    print("\n" + "=" * 75)
    print(" 5. ESM-2 EMBEDDING FEATURE AUDIT")
    print("=" * 75)

    val_h_a_norms = [c["esm_norm_a"] for c in per_val_complex_stats]
    val_h_b_norms = [c["esm_norm_b"] for c in per_val_complex_stats]
    ycr_h_a_norm = float(torch.norm(ycr_h_a, dim=-1).mean().item())
    ycr_h_b_norm = float(torch.norm(ycr_h_b, dim=-1).mean().item())

    print(f"Val Protein A Mean ESM Norm  : {np.mean(val_h_a_norms):.4f} (Min: {np.min(val_h_a_norms):.4f}, Max: {np.max(val_h_a_norms):.4f})")
    print(f"Val Protein B Mean ESM Norm  : {np.mean(val_h_b_norms):.4f} (Min: {np.min(val_h_b_norms):.4f}, Max: {np.max(val_h_b_norms):.4f})")
    print(f"1YCR MDM2 (Chain A) ESM Norm : {ycr_h_a_norm:.4f}")
    print(f"1YCR TP53 (Chain B) ESM Norm : {ycr_h_b_norm:.4f}")

    # 6. Mechanism audit: Why 1D importance r_b has Phe19 high, but 2D contact map precision is diffuse
    print("\n" + "=" * 75)
    print(" 6. 1D VS 2D AGGREGATION & NORMALIZATION MECHANISM AUDIT")
    print("=" * 75)

    # Let's inspect raw attn_matrix, gate_matrix, self_imp_b, max_b, mean_b, r_b_raw, r_b_norm
    with torch.no_grad():
        attn_ab, attn_ba, attn_matrix = model.cross_attn(ycr_h_a, ycr_h_b)
        gate_matrix = model.graph_gating(ycr_h_a, ycr_h_b, ycr_z_a, ycr_z_b)
        interaction_matrix = attn_matrix * gate_matrix
        self_imp_b = model.importance_head(ycr_h_b).squeeze(-1)
        max_b, _ = torch.max(interaction_matrix, dim=0)
        mean_b = torch.mean(interaction_matrix, dim=0)
        r_b_raw = (model.alpha * max_b + (1.0 - model.alpha) * mean_b) * self_imp_b
        r_b_smooth = model.smoother(r_b_raw)
        
        def min_max_norm(x):
            return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)
        
        r_b_norm = min_max_norm(r_b_smooth)

    seq_b = str(ycr_data["seq_b"])
    print(f"{'Residue':<8} | {'p53 #':<6} | {'self_imp_b':<12} | {'max_b (2D)':<12} | {'mean_b (2D)':<12} | {'r_b_raw':<12} | {'r_b_norm':<12} | {'GT Contacts':<12}")
    print("-" * 95)
    for j in range(len(seq_b)):
        aa = seq_b[j]
        p53_num = j + 17
        gt_cnt = int(np.sum(ycr_cmap[:, j]))
        print(f"{aa+str(p53_num):<8} | {p53_num:<6d} | {self_imp_b[j].item():<12.6f} | {max_b[j].item():<12.6f} | {mean_b[j].item():<12.6f} | {r_b_raw[j].item():<12.6f} | {r_b_norm[j].item():<12.6f} | {gt_cnt:<12d}")

    print("\n" + "=" * 75)
    print("                       END OF DIAGNOSTIC")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_forensic_diagnostic()
