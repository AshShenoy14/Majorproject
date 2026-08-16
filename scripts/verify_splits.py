import pandas as pd
import os
import sys
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.utils.paths import PROCESSED_DATA_DIR

def canonical_pair(p1, p2):
    return (p1, p2) if p1 < p2 else (p2, p1)

def verify_splits():
    print("=" * 60)
    print("      DATASET SPLIT & NEGATIVE SAMPLING VERIFICATION")
    print("=" * 60)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path   = PROCESSED_DATA_DIR / "val.csv"
    test_path  = PROCESSED_DATA_DIR / "test.csv"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        print("ERROR: One or more dataset files (train.csv, val.csv, test.csv) missing.")
        return False

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    print(f"\n--- Dataset Sizes ---")
    print(f"Train: {len(train_df)} rows (Pos: {(train_df['label']==1).sum()}, Neg: {(train_df['label']==0).sum()})")
    print(f"Val:   {len(val_df)} rows (Pos: {(val_df['label']==1).sum()}, Neg: {(val_df['label']==0).sum()})")
    print(f"Test:  {len(test_df)} rows (Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()})")

    # 1. Convert to canonical sets
    train_pairs = [canonical_pair(r['protein1'], r['protein2']) for _, r in train_df.iterrows()]
    val_pairs   = [canonical_pair(r['protein1'], r['protein2']) for _, r in val_df.iterrows()]
    test_pairs  = [canonical_pair(r['protein1'], r['protein2']) for _, r in test_df.iterrows()]

    train_set = set(train_pairs)
    val_set   = set(val_pairs)
    test_set  = set(test_pairs)

    # 2. Check internal duplicates
    train_dups = len(train_pairs) - len(train_set)
    val_dups   = len(val_pairs) - len(val_set)
    test_dups  = len(test_pairs) - len(test_set)

    print(f"\n--- Internal Canonical Duplicates ---")
    print(f"Train internal duplicates: {train_dups}")
    print(f"Val internal duplicates:   {val_dups}")
    print(f"Test internal duplicates:  {test_dups}")

    # 3. Check Cross-Split Overlaps
    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    print(f"\n--- Cross-Split Overlaps (Canonical Pairs) ---")
    print(f"Train AND Validation: {train_val_overlap}")
    print(f"Train AND Test:       {train_test_overlap}")
    print(f"Validation AND Test:  {val_test_overlap}")

    # 4. Check Positives vs Negatives Integrity
    all_dfs = pd.concat([train_df, val_df, test_df], ignore_index=True)
    pos_pairs = set(canonical_pair(r['protein1'], r['protein2']) for _, r in all_dfs[all_dfs['label'] == 1].iterrows())
    neg_pairs = set(canonical_pair(r['protein1'], r['protein2']) for _, r in all_dfs[all_dfs['label'] == 0].iterrows())

    pos_neg_leak = len(pos_pairs.intersection(neg_pairs))
    print(f"\n--- Positive / Negative Integrity ---")
    print(f"Total Unique Positive Pairs: {len(pos_pairs)}")
    print(f"Total Unique Negative Pairs: {len(neg_pairs)}")
    print(f"Negative pairs that are known Positives: {pos_neg_leak}")

    # Final summary
    total_issues = train_dups + val_dups + test_dups + train_val_overlap + train_test_overlap + val_test_overlap + pos_neg_leak
    print(f"\n" + "=" * 60)
    if total_issues == 0:
        print("VERIFICATION SUCCESS: 0 overlaps / duplicates found! Split is clean.")
    else:
        print(f"VERIFICATION FAILED: Found {total_issues} total issues/overlaps to resolve.")
    print("=" * 60)

    return total_issues == 0

if __name__ == "__main__":
    verify_splits()
