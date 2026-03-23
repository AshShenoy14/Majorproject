"""
Protein-based dataset splitting with class balancing.
Ensures test proteins are NEVER seen during training.
Handles imbalanced datasets by undersampling the majority class.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.paths import PROCESSED_DATA_DIR


def balance_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Undersample the majority class so pos:neg ≈ 1:1.
    Preserves ALL samples from the minority class.
    """
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    n_minority = min(len(pos), len(neg))

    if n_minority == 0:
        print("  ⚠ No samples from one class — skipping balancing")
        return df

    if len(pos) > len(neg):
        pos_sampled = pos.sample(n=n_minority, random_state=seed)
        balanced = pd.concat([pos_sampled, neg], ignore_index=True)
    else:
        neg_sampled = neg.sample(n=n_minority, random_state=seed)
        balanced = pd.concat([pos, neg_sampled], ignore_index=True)

    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def protein_based_split(
    csv_path: str,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    do_balance: bool = True,
):
    """
    Split interactions so that test proteins NEVER appear in train.
    
    Three categories of interactions:
    - Train: both proteins in train set
    - Val: at least one protein in val set (none in test set)
    - Test: at least one protein in test set
    
    If do_balance=True, each split is undersampled to 1:1 pos:neg ratio.
    This guarantees the model must GENERALIZE to unseen proteins.
    """
    np.random.seed(seed)
    
    df = pd.read_csv(csv_path)
    print(f"Total interactions: {len(df)}")
    print(f"  Positives: {(df['label']==1).sum()}, Negatives: {(df['label']==0).sum()}")
    
    # Collect all unique proteins
    all_proteins = sorted(
        set(df['protein1'].unique()) | set(df['protein2'].unique())
    )
    print(f"Unique proteins: {len(all_proteins)}")
    
    # Compute protein degree (number of interactions)
    degree = defaultdict(int)
    for _, row in df.iterrows():
        degree[row['protein1']] += 1
        degree[row['protein2']] += 1
    
    # Shuffle proteins
    np.random.shuffle(all_proteins)
    
    n_test = int(len(all_proteins) * test_frac)
    n_val = int(len(all_proteins) * val_frac)
    
    test_proteins = set(all_proteins[:n_test])
    val_proteins = set(all_proteins[n_test:n_test + n_val])
    train_proteins = set(all_proteins[n_test + n_val:])
    
    print(f"Protein split: {len(train_proteins)} train / "
          f"{len(val_proteins)} val / {len(test_proteins)} test")
    
    # Assign each interaction to a split
    train_rows, val_rows, test_rows = [], [], []
    
    for _, row in df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        
        # If EITHER protein is in test set → test interaction
        if p1 in test_proteins or p2 in test_proteins:
            test_rows.append(row)
        # If EITHER protein is in val set → val interaction
        elif p1 in val_proteins or p2 in val_proteins:
            val_rows.append(row)
        # Both proteins are train-only
        else:
            train_rows.append(row)
    
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    
    # Verify no protein leakage
    train_p = set(train_df['protein1']) | set(train_df['protein2'])
    test_p = set(test_df['protein1']) | set(test_df['protein2'])
    leaked = train_p & test_proteins
    
    print(f"\n--- Before balancing ---")
    print(f"  Train: {len(train_df)} (pos={( train_df['label']==1).sum()}, neg={(train_df['label']==0).sum()})")
    print(f"  Val:   {len(val_df)} (pos={(val_df['label']==1).sum()}, neg={(val_df['label']==0).sum()})")
    print(f"  Test:  {len(test_df)} (pos={(test_df['label']==1).sum()}, neg={(test_df['label']==0).sum()})")
    print(f"  Protein leakage: {len(leaked)} "
          f"({'CLEAN' if len(leaked) == 0 else 'LEAKED!'})")
    
    # Balance each split
    if do_balance:
        print(f"\n--- Balancing (undersampling majority class) ---")
        train_df = balance_split(train_df, seed=seed)
        val_df = balance_split(val_df, seed=seed + 1)
        test_df = balance_split(test_df, seed=seed + 2)
        
        print(f"\n--- After balancing ---")
        print(f"  Train: {len(train_df)} (pos={(train_df['label']==1).sum()}, neg={(train_df['label']==0).sum()})")
        print(f"  Val:   {len(val_df)} (pos={(val_df['label']==1).sum()}, neg={(val_df['label']==0).sum()})")
        print(f"  Test:  {len(test_df)} (pos={(test_df['label']==1).sum()}, neg={(test_df['label']==0).sum()})")
    else:
        print("\n--- Balancing SKIPPED (--no-balance flag) ---")
    
    # Save
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    # Save protein sets for later verification
    import torch
    torch.save({
        'train_proteins': train_proteins,
        'val_proteins': val_proteins,
        'test_proteins': test_proteins,
    }, PROCESSED_DATA_DIR / "protein_splits.pt")
    
    print(f"\nSaved to {PROCESSED_DATA_DIR}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein-based dataset splitting")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip undersampling (keep imbalanced data)")
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args()

    # Re-split your existing combined data
    combined_path = PROCESSED_DATA_DIR / "interactions_with_negatives.csv"
    
    if not combined_path.exists():
        # Reconstruct from existing splits
        dfs = []
        for f in ["train.csv", "val.csv", "test.csv"]:
            p = PROCESSED_DATA_DIR / f
            if p.exists():
                dfs.append(pd.read_csv(p))
        if dfs:
            combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
            combined.to_csv(combined_path, index=False)
            print(f"Combined {len(combined)} interactions → {combined_path}")
        else:
            print("No existing splits to combine!")
            sys.exit(1)
    
    protein_based_split(
        combined_path,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        do_balance=not args.no_balance,
    )
