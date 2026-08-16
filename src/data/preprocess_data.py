import pandas as pd
import numpy as np
import os
import sys
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Set

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, STRING_FILE, UNIPROT_FILE

def load_string_interactions(filepath: str, min_score: int = 700) -> pd.DataFrame:
    """
    Loads STRING interactions, filtering by confidence score.
    Enforces canonical ordering (protein1 < protein2) and removes duplicates.
    STRING format: protein1 protein2 score ...
    IDs are usually '9606.ENSP...'
    """
    print(f"Loading interactions from {filepath}...")
    df = pd.read_csv(filepath, sep=" ", usecols=["protein1", "protein2", "combined_score"])
    
    # Filter by score (STRING score is 0-1000)
    df = df[df["combined_score"] >= min_score].copy()
    
    # Clean IDs: remove '9606.' prefix if present
    df["protein1"] = df["protein1"].astype(str).str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].astype(str).str.replace("9606.", "", regex=False)
    
    # Remove self-loops
    df = df[df["protein1"] != df["protein2"]].copy()
    
    # Enforce canonical order (protein1 < protein2)
    p1 = np.where(df["protein1"] < df["protein2"], df["protein1"], df["protein2"])
    p2 = np.where(df["protein1"] < df["protein2"], df["protein2"], df["protein1"])
    
    canonical_df = pd.DataFrame({"protein1": p1, "protein2": p2}).drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)
    print(f"Loaded {len(canonical_df)} canonical unique positive interactions.")
    return canonical_df

def load_sequences(fasta_file: str) -> dict:
    """
    Loads sequences from a FASTA file into a dictionary.
    Keys should match the IDs in the interaction file.
    """
    print(f"Loading sequences from {fasta_file}...")
    seqs = {}
    if not os.path.exists(fasta_file):
        print(f"Warning: Sequence file {fasta_file} not found.")
        return {}

    if str(fasta_file).endswith(".gz"):
        import gzip
        handle = gzip.open(fasta_file, "rt")
    else:
        handle = open(fasta_file, "r")

    for record in SeqIO.parse(handle, "fasta"):
        seq_id = record.id.split("|")[1] if "|" in record.id else record.id
        seqs[seq_id] = str(record.seq)
    handle.close()
    return seqs

def load_localization_cache() -> dict:
    """
    Loads biological localization cache as a dictionary of protein_id -> set of locations.
    """
    loc_dict = {}
    try:
        cache_path = PROCESSED_DATA_DIR / "bio_metadata_cache.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path).fillna("")
            for _, row in df.iterrows():
                pid = str(row.get("protein_id", "")).strip()
                loc_str = str(row.get("localization", "")).strip()
                if pid and loc_str:
                    locs = {x.strip().lower() for x in loc_str.split(";") if x.strip()}
                    if locs:
                        loc_dict[pid] = locs
            print(f"Loaded {len(loc_dict)} protein localizations from cache.")
    except Exception as e:
        print(f"Warning: Could not load localization cache ({e})")
    return loc_dict

def is_co_localized(p1: str, p2: str, loc_dict: dict) -> bool:
    """
    Returns True if p1 and p2 share at least one subcellular localization.
    If either protein has no localization data, defaults to True to prevent over-filtering.
    """
    if not loc_dict:
        return True
    locs1 = loc_dict.get(p1)
    locs2 = loc_dict.get(p2)
    if not locs1 or not locs2:
        return True
    return len(locs1.intersection(locs2)) > 0

def generate_hard_negatives(positive_df: pd.DataFrame, all_proteins: List[str], ratio: float = 0.5, loc_dict: dict = None) -> pd.DataFrame:
    """
    Generates 'hard' negative samples using Common Neighbors strategy.
    Pairs that share neighbors but don't interact are harder to distinguish.
    Filtered by subcellular co-localization if loc_dict is provided.
    Returns canonical unique negative pairs.
    """
    print("Generating hard negative samples (Common Neighbors)...")
    from collections import defaultdict
    adj = defaultdict(set)
    for u, v in zip(positive_df["protein1"], positive_df["protein2"]):
        adj[u].add(v)
        adj[v].add(u)
    
    positive_pairs = set(zip(positive_df["protein1"], positive_df["protein2"]))
    hard_negatives = set()
    num_needed = int(len(positive_df) * ratio)
    
    p_list = list(adj.keys())
    np.random.seed(42)
    np.random.shuffle(p_list)
    
    for u in p_list:
        if len(hard_negatives) >= num_needed:
            break
        
        neighbors = adj[u]
        for neighbor in neighbors:
            for v in adj[neighbor]:
                if u == v:
                    continue
                
                u_c, v_c = (u, v) if u < v else (v, u)
                
                if (u_c, v_c) not in positive_pairs and (u_c, v_c) not in hard_negatives:
                    if loc_dict is None or is_co_localized(u_c, v_c, loc_dict):
                        hard_negatives.add((u_c, v_c))
                        if len(hard_negatives) >= num_needed:
                            break
            if len(hard_negatives) >= num_needed:
                break
            
    print(f"Generated {len(hard_negatives)} hard negatives.")
    if hard_negatives:
        return pd.DataFrame(list(hard_negatives), columns=["protein1", "protein2"])
    else:
        return pd.DataFrame(columns=["protein1", "protein2"])

def generate_negative_samples(positive_df: pd.DataFrame, all_proteins: List[str], ratio: float = 1.0, hard_ratio: float = 0.5, loc_dict: dict = None) -> pd.DataFrame:
    """
    Generates negative samples by mixing random (easy) and common-neighbor (hard) pairs.
    All pairs are canonicalized (protein1 < protein2) and strictly deduplicated.
    """
    total_needed = int(len(positive_df) * ratio)
    
    hard_df = generate_hard_negatives(positive_df, all_proteins, ratio=hard_ratio, loc_dict=loc_dict)
    num_hard = len(hard_df)
    num_easy = total_needed - num_hard
    
    print(f"Generating {num_easy} easy (random) negative samples with localization constraints...")
    positive_pairs = set(zip(positive_df["protein1"], positive_df["protein2"]))
    hard_pairs = set(zip(hard_df["protein1"], hard_df["protein2"])) if not hard_df.empty else set()
    negative_pairs = set()
    
    all_proteins_arr = np.array(all_proteins)
    np.random.seed(42)
    max_attempts = num_easy * 100
    attempts = 0
    
    batch_size = max(10000, num_easy * 2)
    while len(negative_pairs) < num_easy and attempts < max_attempts:
        p1 = np.random.choice(all_proteins_arr, batch_size, replace=True)
        p2 = np.random.choice(all_proteins_arr, batch_size, replace=True)
        
        for u, v in zip(p1, p2):
            attempts += 1
            if u == v:
                continue
            u_c, v_c = (u, v) if u < v else (v, u)
            
            if (u_c, v_c) not in positive_pairs and (u_c, v_c) not in hard_pairs and (u_c, v_c) not in negative_pairs:
                if loc_dict is None or is_co_localized(u_c, v_c, loc_dict):
                    negative_pairs.add((u_c, v_c))
                    if len(negative_pairs) >= num_easy:
                        break
                        
    easy_df = pd.DataFrame(list(negative_pairs), columns=["protein1", "protein2"])
    full_neg_df = pd.concat([easy_df, hard_df], ignore_index=True).drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)
    print(f"Total unique negative samples generated: {len(full_neg_df)}")
    return full_neg_df

def preprocess_data(min_score: int = 900, hard_ratio: float = 0.5):
    # 1. Load Interactions
    if not STRING_FILE.exists():
        print(f"Error: {STRING_FILE} not found. Please run collect_ppi.py first.")
        return

    inte_df = load_string_interactions(STRING_FILE, min_score=min_score)
    
    # 2. Get unique proteins
    unique_proteins = sorted(list(set(inte_df["protein1"]).union(set(inte_df["protein2"]))))
    print(f"Found {len(unique_proteins)} unique proteins in positive set.")
    
    # 3. Load localization cache
    loc_dict = load_localization_cache()
    
    # 4. Generate Negatives
    neg_df = generate_negative_samples(inte_df, unique_proteins, ratio=1.0, hard_ratio=hard_ratio, loc_dict=loc_dict)
    
    # 5. Labeling
    inte_df["label"] = 1
    neg_df["label"] = 0
    
    # 6. Combine & Final Deduplication Check
    full_df = pd.concat([inte_df, neg_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)
    
    print(f"Full dataset size: {len(full_df)} (Positives: {(full_df['label'] == 1).sum()}, Negatives: {(full_df['label'] == 0).sum()})")
    
    # 7. Leak-Free Pair Splitting
    train_df, test_temp_df = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
    val_df, test_df = train_test_split(test_temp_df, test_size=0.5, stratify=test_temp_df["label"], random_state=42)
    
    # 8. Save
    print("Saving processed clean dataset...")
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_score", type=int, default=900)
    parser.add_argument("--hard_ratio", type=float, default=0.5)
    args = parser.parse_args()
    preprocess_data(min_score=args.min_score, hard_ratio=args.hard_ratio)

