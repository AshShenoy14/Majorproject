"""
Script to process Supp-B.doc negative PPI dataset:
1. Extract negative pairs (UniProt IDs) from the .doc file
2. Map UniProt IDs to ENSP IDs using the idmapping file
3. Replace existing synthesized negatives in train/test/val with real negatives
4. Maintain the same CSV format: protein1,protein2,label
"""

import olefile
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
DOC_PATH = r"C:\Users\bbbba\Downloads\Supp-B.doc"
MAPPING_PATH = r"c:\Majorproject\data\raw\idmapping_2026_02_16.tsv"
DATA_DIR = Path(r"c:\Majorproject\data\processed")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
VAL_PATH = DATA_DIR / "val.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions_with_negatives.csv"

# ─── Step 1: Extract negative pairs from .doc file ──────────────────────────
print("=" * 60)
print("Step 1: Extracting negative pairs from Supp-B.doc")
print("=" * 60)

ole = olefile.OleFileIO(DOC_PATH)
data = ole.openstream('WordDocument').read()
text = data.decode('latin-1', errors='ignore')

# Extract UniProt ID pairs (format: ID1 whitespace ID2)
# UniProt IDs: [A-Z][0-9][A-Z0-9]{3}[0-9] (e.g., Q9BWT1, P14598)
pairs_raw = re.findall(
    r'([A-Z][0-9][A-Z0-9]{3}[0-9])\s+([A-Z][0-9][A-Z0-9]{3}[0-9])',
    text
)

# Deduplicate (order-independent: (A,B) == (B,A))
seen = set()
unique_pairs = []
for p1, p2 in pairs_raw:
    key = tuple(sorted([p1, p2]))
    if key not in seen:
        seen.add(key)
        unique_pairs.append((p1, p2))

print(f"  Raw pairs extracted: {len(pairs_raw)}")
print(f"  Unique pairs (deduplicated): {len(unique_pairs)}")
print(f"  Sample pairs: {unique_pairs[:5]}")

# Collect all unique UniProt IDs
all_uniprot_ids = set()
for p1, p2 in unique_pairs:
    all_uniprot_ids.add(p1)
    all_uniprot_ids.add(p2)
print(f"  Unique UniProt IDs: {len(all_uniprot_ids)}")

# ─── Step 2: Build UniProt → ENSP mapping ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Building UniProt → ENSP mapping")
print("=" * 60)

mapping_df = pd.read_csv(MAPPING_PATH, sep='\t', header=0)
print(f"  Mapping file columns: {mapping_df.columns.tolist()}")
print(f"  Mapping file shape: {mapping_df.shape}")
print(f"  First few rows:")
print(mapping_df.head())

# The mapping file has: From (ENSP), Entry (UniProt), Entry Name
# We need UniProt -> ENSP (reverse mapping)
# Column 'From' = ENSP ID, Column 'Entry' = UniProt ID
uniprot_to_ensp = {}
for _, row in mapping_df.iterrows():
    ensp = str(row.iloc[0]).strip()  # "From" column = ENSP
    uniprot = str(row.iloc[1]).strip()  # "Entry" column = UniProt
    if uniprot in all_uniprot_ids:
        # Strip "9606." prefix if present
        ensp_clean = ensp.replace("9606.", "")
        uniprot_to_ensp[uniprot] = ensp_clean

print(f"\n  Mapped {len(uniprot_to_ensp)} / {len(all_uniprot_ids)} UniProt IDs to ENSP IDs")

# Show unmapped IDs
unmapped = all_uniprot_ids - set(uniprot_to_ensp.keys())
if unmapped:
    print(f"  Unmapped UniProt IDs ({len(unmapped)}): {list(unmapped)[:20]}")

# ─── Step 3: Convert pairs to ENSP format ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Converting pairs to ENSP format")
print("=" * 60)

negative_pairs = []
skipped = 0
for p1, p2 in unique_pairs:
    if p1 in uniprot_to_ensp and p2 in uniprot_to_ensp:
        negative_pairs.append({
            'protein1': uniprot_to_ensp[p1],
            'protein2': uniprot_to_ensp[p2],
            'label': 0
        })
    else:
        skipped += 1

neg_df = pd.DataFrame(negative_pairs)
print(f"  Successfully converted pairs: {len(neg_df)}")
print(f"  Skipped (missing ENSP mapping): {skipped}")
print(f"  Sample converted pairs:")
print(neg_df.head(10))

# ─── Step 4: Load existing data and analyze ─────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Loading existing data")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"  Train: {train_df.shape} | Pos: {(train_df['label']==1).sum()}, Neg: {(train_df['label']==0).sum()}")
print(f"  Test:  {test_df.shape}  | Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()}")
print(f"  Val:   {val_df.shape}  | Pos: {(val_df['label']==1).sum()}, Neg: {(val_df['label']==0).sum()}")

# Count existing positives
train_pos = train_df[train_df['label'] == 1]
test_pos = test_df[test_df['label'] == 1]
val_pos = val_df[val_df['label'] == 1]

total_pos = len(train_pos) + len(test_pos) + len(val_pos)
print(f"\n  Total positives across all splits: {total_pos}")
print(f"  Total new negatives available: {len(neg_df)}")

# ─── Step 5: Replace synthesized negatives with real negatives ───────────────
print("\n" + "=" * 60)
print("Step 5: Replacing synthesized negatives with real negatives")
print("=" * 60)

# Shuffle negatives
neg_df_shuffled = neg_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Determine split ratios based on existing positive split
total_existing = len(train_df) + len(test_df) + len(val_df)
train_ratio = len(train_pos) / total_pos
test_ratio = len(test_pos) / total_pos
val_ratio = len(val_pos) / total_pos

print(f"  Split ratios (from positives) - Train: {train_ratio:.3f}, Test: {test_ratio:.3f}, Val: {val_ratio:.3f}")

# Determine how many negatives to assign to each split
# Try to match the number of positives in each split (1:1 ratio)
# But cap at available negatives
n_neg_available = len(neg_df_shuffled)

# Option 1: Equal to number of positives (balanced)
n_train_neg = min(len(train_pos), int(train_ratio * n_neg_available))
n_test_neg = min(len(test_pos), int(test_ratio * n_neg_available))
n_val_neg = min(len(val_pos), int(val_ratio * n_neg_available))

# Adjust to not exceed available
total_needed = n_train_neg + n_test_neg + n_val_neg
if total_needed > n_neg_available:
    scale = n_neg_available / total_needed
    n_train_neg = int(n_train_neg * scale)
    n_test_neg = int(n_test_neg * scale)
    n_val_neg = n_neg_available - n_train_neg - n_test_neg

print(f"  Negatives allocation - Train: {n_train_neg}, Test: {n_test_neg}, Val: {n_val_neg}")
print(f"  Total negatives used: {n_train_neg + n_test_neg + n_val_neg} / {n_neg_available}")

# Split negatives
train_neg = neg_df_shuffled.iloc[:n_train_neg]
test_neg = neg_df_shuffled.iloc[n_train_neg:n_train_neg + n_test_neg]
val_neg = neg_df_shuffled.iloc[n_train_neg + n_test_neg:n_train_neg + n_test_neg + n_val_neg]

# Combine with positives
new_train = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
new_test = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
new_val = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  New Train: {new_train.shape} | Pos: {(new_train['label']==1).sum()}, Neg: {(new_train['label']==0).sum()}")
print(f"  New Test:  {new_test.shape}  | Pos: {(new_test['label']==1).sum()}, Neg: {(new_test['label']==0).sum()}")
print(f"  New Val:   {new_val.shape}  | Pos: {(new_val['label']==1).sum()}, Neg: {(new_val['label']==0).sum()}")

# ─── Step 6: Back up old data and save new data ─────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Saving data")
print("=" * 60)

# Backup
backup_dir = DATA_DIR / "backup_synthesized"
backup_dir.mkdir(exist_ok=True)

train_df.to_csv(backup_dir / "train_old.csv", index=False)
test_df.to_csv(backup_dir / "test_old.csv", index=False)
val_df.to_csv(backup_dir / "val_old.csv", index=False)
print(f"  Old data backed up to: {backup_dir}")

# Save new data
new_train.to_csv(TRAIN_PATH, index=False)
new_test.to_csv(TEST_PATH, index=False)
new_val.to_csv(VAL_PATH, index=False)
print(f"  New train.csv saved: {TRAIN_PATH}")
print(f"  New test.csv saved: {TEST_PATH}")
print(f"  New val.csv saved: {VAL_PATH}")

# Also update interactions_with_negatives.csv
all_new = pd.concat([new_train, new_test, new_val], ignore_index=True)
all_new.to_csv(INTERACTIONS_PATH, index=False)
print(f"  New interactions_with_negatives.csv saved: {INTERACTIONS_PATH}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Negative pairs from Supp-B.doc: {len(unique_pairs)}")
print(f"  Successfully mapped to ENSP: {len(neg_df)}")
print(f"  Old synthesized negatives removed")
print(f"  New real negatives distributed across train/test/val")
print(f"  Data format: protein1,protein2,label (same as before)")
print(f"  All negative labels = 0, all positive labels = 1")
