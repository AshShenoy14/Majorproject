import os
import sys
import glob
import json
import random
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def perform_dataset_audit():
    dataset_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    data_files = sorted(glob.glob(str(dataset_dir / "*.npz")))
    
    total_complexes = len(data_files)
    valid_complexes = 0
    failed_complexes = 0
    total_pairs = 0
    total_contacts = 0
    zero_contact_complexes = 0
    dim_mismatches = 0
    invalid_binary_count = 0
    inf_nan_embedding_count = 0
    
    file_details = []

    for f_path in data_files:
        try:
            data = np.load(f_path)
            seq_a = str(data["seq_a"])
            seq_b = str(data["seq_b"])
            emb_a = data["esm_embedding_a"]
            emb_b = data["esm_embedding_b"]
            cmap = data["contact_map"]
            l_a = int(data["length_a"])
            l_b = int(data["length_b"])

            is_valid = True
            
            # Check dimensions
            if len(seq_a) != emb_a.shape[0] or len(seq_b) != emb_b.shape[0] or cmap.shape != (l_a, l_b):
                dim_mismatches += 1
                is_valid = False

            # Check binary labels
            unique_vals = set(np.unique(cmap))
            if not unique_vals.issubset({0, 1}):
                invalid_binary_count += 1
                is_valid = False

            # Check Inf/NaN
            if np.isnan(emb_a).any() or np.isinf(emb_a).any() or np.isnan(emb_b).any() or np.isinf(emb_b).any():
                inf_nan_embedding_count += 1
                is_valid = False

            n_contacts = int(cmap.sum())
            n_pairs = cmap.size
            if n_contacts == 0:
                zero_contact_complexes += 1

            if is_valid:
                valid_complexes += 1
                total_contacts += n_contacts
                total_pairs += n_pairs
            else:
                failed_complexes += 1

            file_details.append({
                "file": Path(f_path).name,
                "complex_id": str(data.get("complex_id", "")),
                "length_a": l_a,
                "length_b": l_b,
                "total_pairs": n_pairs,
                "contacts": n_contacts,
                "valid": is_valid
            })
        except Exception as e:
            failed_complexes += 1
            print(f"Error auditing file {f_path}: {e}")

    # Train/Val Split Audit
    seed = 42
    rng = random.Random(seed)
    shuffled_files = list(data_files)
    rng.shuffle(shuffled_files)

    val_size = max(1, int(len(shuffled_files) * 0.2))
    val_files = shuffled_files[:val_size]
    train_files = shuffled_files[val_size:]

    def get_split_stats(file_list):
        s_contacts = 0
        s_pairs = 0
        for f in file_list:
            d = np.load(f)
            cm = d["contact_map"]
            s_contacts += int(cm.sum())
            s_pairs += cm.size
        mean_c = s_contacts / len(file_list) if file_list else 0.0
        sparsity = (1.0 - (s_contacts / s_pairs)) * 100.0 if s_pairs > 0 else 0.0
        return len(file_list), s_pairs, s_contacts, mean_c, sparsity

    tr_count, tr_pairs, tr_contacts, tr_mean, tr_sparsity = get_split_stats(train_files)
    val_count, val_pairs, val_contacts, val_mean, val_sparsity = get_split_stats(val_files)

    # Verify zero pair overlap between train and val
    train_set_names = set(Path(f).name for f in train_files)
    val_set_names = set(Path(f).name for f in val_files)
    overlap = len(train_set_names.intersection(val_set_names)) == 0

    mean_contacts = total_contacts / valid_complexes if valid_complexes > 0 else 0.0
    overall_sparsity = (1.0 - (total_contacts / total_pairs)) * 100.0 if total_pairs > 0 else 0.0

    report = {
        "total_complexes": total_complexes,
        "valid_complexes": valid_complexes,
        "failed_complexes": failed_complexes,
        "total_residue_pairs": total_pairs,
        "total_contacts": total_contacts,
        "mean_contacts_per_complex": mean_contacts,
        "contact_sparsity_percent": overall_sparsity,
        "zero_contact_complexes": zero_contact_complexes,
        "dimension_mismatch_count": dim_mismatches,
        "invalid_binary_labels_count": invalid_binary_count,
        "inf_nan_embedding_count": inf_nan_embedding_count,
        "train_val_split": {
            "random_seed": seed,
            "train_complexes": tr_count,
            "val_complexes": val_count,
            "train_total_pairs": tr_pairs,
            "val_total_pairs": val_pairs,
            "train_contacts": tr_contacts,
            "val_contacts": val_contacts,
            "train_mean_contacts": tr_mean,
            "val_mean_contacts": val_mean,
            "train_contact_sparsity_percent": tr_sparsity,
            "val_contact_sparsity_percent": val_sparsity,
            "zero_pair_overlap": overlap
        }
    }

    report_path = dataset_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n==========================================================================")
    print("                    IRLM DATASET QUALITY AUDIT REPORT                     ")
    print("==========================================================================")
    print(f" Total PDB Complexes           : {total_complexes}")
    print(f" Valid Complexes               : {valid_complexes}")
    print(f" Failed Complexes              : {failed_complexes}")
    print(f" Total Residue Pairs           : {total_pairs:,}")
    print(f" Total Positive Contacts (<8A) : {total_contacts:,}")
    print(f" Mean Contacts per Complex     : {mean_contacts:.2f}")
    print(f" Overall Contact Sparsity      : {overall_sparsity:.4f}%")
    print(f" Zero-Contact Complexes        : {zero_contact_complexes}")
    print(f" Dimension Mismatches          : {dim_mismatches}")
    print(f" Invalid Binary Labels         : {invalid_binary_count}")
    print(f" Inf/NaN Embeddings            : {inf_nan_embedding_count}")
    print("--------------------------------------------------------------------------")
    print(" TRAIN/VAL SPLIT VERIFICATION (Seed 42, 80/20 Split):")
    print(f"   Train Set                   : {tr_count} complexes, {tr_pairs:,} pairs, {tr_contacts:,} contacts ({tr_mean:.2f} mean/complex)")
    print(f"   Val Set                     : {val_count} complexes, {val_pairs:,} pairs, {val_contacts:,} contacts ({val_mean:.2f} mean/complex)")
    print(f"   Zero Pair Overlap           : {'VERIFIED (TRUE)' if overlap else 'FAILED (FALSE)'}")
    print(f" Saved Audit Report            : {report_path}")
    print("==========================================================================\n")

if __name__ == "__main__":
    perform_dataset_audit()
