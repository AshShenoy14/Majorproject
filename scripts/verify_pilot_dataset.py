import os
import sys
import csv
import glob
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def verify_dataset():
    data_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    manifest_path = data_dir / "manifest.csv"

    print("=" * 70)
    print("        VERIFYING IRLM PILOT STRUCTURAL DATASET ARTIFACTS")
    print("=" * 70)

    if not data_dir.exists():
        print(f"ERROR: Dataset directory {data_dir} does not exist!")
        return False

    npz_files = sorted(glob.glob(str(data_dir / "*.npz")))
    print(f"Found {len(npz_files)} .npz files in {data_dir}")

    if not manifest_path.exists():
        print(f"ERROR: Manifest file {manifest_path} does not exist!")
        return False

    # 1. Inspect manifest
    manifest_entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_entries.append(row)

    print(f"Manifest contains {len(manifest_entries)} total entries.")
    successful_manifest = [r for r in manifest_entries if r["status"] == "SUCCESS"]
    print(f"Manifest successful entries: {len(successful_manifest)}")

    # 2. Detailed audit of each .npz file
    passed_files = 0
    failed_files = 0
    
    seq_lengths_a = []
    seq_lengths_b = []
    contact_counts = []
    contact_densities = []

    expected_keys = {
        'complex_id', 'seq_a', 'seq_b', 'length_a', 'length_b',
        'esm_embedding_a', 'esm_embedding_b', 'contact_map',
        'interface_mask_a', 'interface_mask_b'
    }

    for filepath in npz_files:
        filename = os.path.basename(filepath)
        try:
            data = np.load(filepath, allow_pickle=True)
            
            # Key presence check
            keys = set(data.files)
            if not expected_keys.issubset(keys):
                missing = expected_keys - keys
                print(f"  [FAIL] {filename}: Missing keys {missing}")
                failed_files += 1
                continue

            seq_a = str(data['seq_a'])
            seq_b = str(data['seq_b'])
            L_A = int(data['length_a'])
            L_B = int(data['length_b'])
            emb_a = data['esm_embedding_a']
            emb_b = data['esm_embedding_b']
            cmap = data['contact_map']
            mask_a = data['interface_mask_a']
            mask_b = data['interface_mask_b']

            # 1. Length consistency
            if len(seq_a) != L_A or emb_a.shape[0] != L_A or cmap.shape[0] != L_A or mask_a.shape[0] != L_A:
                print(f"  [FAIL] {filename}: Chain A dimension mismatch! L_A={L_A}, seq_len={len(seq_a)}, emb_shape={emb_a.shape}, cmap_shape={cmap.shape}, mask_shape={mask_a.shape}")
                failed_files += 1
                continue

            if len(seq_b) != L_B or emb_b.shape[0] != L_B or cmap.shape[1] != L_B or mask_b.shape[0] != L_B:
                print(f"  [FAIL] {filename}: Chain B dimension mismatch! L_B={L_B}, seq_len={len(seq_b)}, emb_shape={emb_b.shape}, cmap_shape={cmap.shape}, mask_shape={mask_b.shape}")
                failed_files += 1
                continue

            # 2. Embedding dimension
            if emb_a.shape[1] != 480 or emb_b.shape[1] != 480:
                print(f"  [FAIL] {filename}: Embedding feature dim is not 480! emb_a={emb_a.shape}, emb_b={emb_b.shape}")
                failed_files += 1
                continue

            # 3. NaNs / Infs
            if np.isnan(emb_a).any() or np.isinf(emb_a).any() or np.isnan(emb_b).any() or np.isinf(emb_b).any():
                print(f"  [FAIL] {filename}: Embeddings contain NaN or Inf values!")
                failed_files += 1
                continue

            if np.isnan(cmap).any() or np.isinf(cmap).any():
                print(f"  [FAIL] {filename}: Contact map contains NaN or Inf values!")
                failed_files += 1
                continue

            # 4. Binary values
            unique_cmap = set(np.unique(cmap))
            if not unique_cmap.issubset({0, 1}):
                print(f"  [FAIL] {filename}: Contact map is not strictly binary {unique_cmap}")
                failed_files += 1
                continue

            num_contacts = int(cmap.sum())
            if num_contacts == 0:
                print(f"  [FAIL] {filename}: Zero contacts in contact map!")
                failed_files += 1
                continue

            # 5. Interface mask verification
            expected_mask_a = (cmap.sum(axis=1) > 0).astype(np.uint8)
            expected_mask_b = (cmap.sum(axis=0) > 0).astype(np.uint8)
            if not np.array_equal(mask_a, expected_mask_a) or not np.array_equal(mask_b, expected_mask_b):
                print(f"  [FAIL] {filename}: Interface masks do not match contact map projections!")
                failed_files += 1
                continue

            density = num_contacts / (L_A * L_B)
            seq_lengths_a.append(L_A)
            seq_lengths_b.append(L_B)
            contact_counts.append(num_contacts)
            contact_densities.append(density)
            passed_files += 1

        except Exception as e:
            print(f"  [FAIL] {filename}: Exception during inspection ({e})")
            failed_files += 1

    print("\n" + "=" * 70)
    print("                  VERIFICATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total .npz Artifacts Inspected : {len(npz_files)}")
    print(f"Passed All Strict Validations  : {passed_files}")
    print(f"Failed Validations              : {failed_files}")
    
    if passed_files > 0:
        print("-" * 70)
        print("DATASET STATISTICAL PROFILE:")
        print(f"  Mean Chain A Length  : {np.mean(seq_lengths_a):.1f} ± {np.std(seq_lengths_a):.1f} (min: {np.min(seq_lengths_a)}, max: {np.max(seq_lengths_a)})")
        print(f"  Mean Chain B Length  : {np.mean(seq_lengths_b):.1f} ± {np.std(seq_lengths_b):.1f} (min: {np.min(seq_lengths_b)}, max: {np.max(seq_lengths_b)})")
        print(f"  Mean Contacts / Pair : {np.mean(contact_counts):.1f} (min: {np.min(contact_counts)}, max: {np.max(contact_counts)})")
        print(f"  Mean Contact Density : {np.mean(contact_densities):.4f} (min: {np.min(contact_densities):.4f}, max: {np.max(contact_densities):.4f})")
        print(f"  ESM Embedding Dim    : 480 (facebook/esm2_t12_35M_UR50D)")
    print("=" * 70)

    return failed_files == 0 and passed_files >= 50

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
