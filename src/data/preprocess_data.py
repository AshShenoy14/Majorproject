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
    STRING format: protein1 protein2 score ...
    IDs are usually '9606.ENSP...'
    """
    print(f"Loading interactions from {filepath}...")
    # STRING files are space-separated
    df = pd.read_csv(filepath, sep=" ", usecols=["protein1", "protein2", "combined_score"])
    
    # Filter by score (STRING score is 0-1000)
    df = df[df["combined_score"] >= min_score]
    
    # Clean IDs: remove '9606.' prefix if present
    df["protein1"] = df["protein1"].str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace("9606.", "", regex=False)
    
    return df[["protein1", "protein2"]]

def load_sequences(fasta_file: str) -> dict:
    """
    Loads sequences from a FASTA file into a dictionary.
    Keys should match the IDs in the interaction file.
    Note: STRING uses Ensembl protein IDs (ENSP). UniProt fasta might have UniProt IDs.
    Mapping might be needed. For this template, we assume IDs match or we use a mapping.
    """
    print(f"Loading sequences from {fasta_file}...")
    seqs = {}
    # This is a placeholder. In reality, we need ID mapping ENSP <-> UniProt
    # Or we download the sequences from STRING which uses ENSP IDs.
    if not os.path.exists(fasta_file):
        print(f"Warning: Sequence file {fasta_file} not found.")
        return {}

    if str(fasta_file).endswith(".gz"):
        import gzip
        handle = gzip.open(fasta_file, "rt")
    else:
        handle = open(fasta_file, "r")

    for record in SeqIO.parse(handle, "fasta"):
        # We need to extract the ID that matches the interaction file.
        # This is highly dependent on the source file headers.
        seq_id = record.id.split("|")[1] if "|" in record.id else record.id
        seqs[seq_id] = str(record.seq)
    return seqs

def generate_hard_negatives(positive_df: pd.DataFrame, all_proteins: List[str], ratio: float = 0.5) -> pd.DataFrame:
    """
    Generates 'hard' negative samples using Common Neighbors strategy.
    Pairs that share neighbors but don't interact are harder to distinguish.
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
    
    # Stratified sampling: for each protein, look at its neighbors' neighbors
    p_list = list(adj.keys())
    np.random.shuffle(p_list)
    
    for u in p_list:
        if len(hard_negatives) >= num_needed: break
        
        neighbors = adj[u]
        for neighbor in neighbors:
            # Look at neighbor's neighbors (potential hard negatives for u)
            for v in adj[neighbor]:
                if u == v: continue
                
                # Canonical order
                u_c, v_c = (u, v) if u < v else (v, u)
                
                if (u_c, v_c) not in positive_pairs and (u_c, v_c) not in hard_negatives:
                    hard_negatives.add((u_c, v_c))
                    if len(hard_negatives) >= num_needed:
                        break
            if len(hard_negatives) >= num_needed: break
            
    print(f"Generated {len(hard_negatives)} hard negatives.")
    return pd.DataFrame(list(hard_negatives), columns=["protein1", "protein2"])

def generate_negative_samples(positive_df: pd.DataFrame, all_proteins: List[str], ratio: float = 1.0, hard_ratio: float = 0.5) -> pd.DataFrame:
    """
    Generates negative samples by mixing random (easy) and common-neighbor (hard) pairs.
    """
    num_hard = int(len(positive_df) * ratio * hard_ratio)
    num_easy = int(len(positive_df) * ratio) - num_hard
    
    hard_df = generate_hard_negatives(positive_df, all_proteins, ratio=hard_ratio)
    
    print(f"Generating {num_easy} easy (random) negative samples...")
    positive_pairs = set(zip(positive_df["protein1"], positive_df["protein2"]))
    hard_pairs = set(zip(hard_df["protein1"], hard_df["protein2"]))
    negative_pairs = set()
    
    all_proteins_arr = np.array(all_proteins)
    while len(negative_pairs) < num_easy:
        p1 = np.random.choice(all_proteins_arr, num_easy, replace=True)
        p2 = np.random.choice(all_proteins_arr, num_easy, replace=True)
        
        for u, v in zip(p1, p2):
            if u == v: continue
            if u > v: u, v = v, u
            
            if (u, v) not in positive_pairs and (u, v) not in hard_pairs and (u, v) not in negative_pairs:
                negative_pairs.add((u, v))
                if len(negative_pairs) >= num_easy:
                    break
                    
    easy_df = pd.DataFrame(list(negative_pairs), columns=["protein1", "protein2"])
    return pd.concat([easy_df, hard_df], ignore_index=True)

def preprocess_data(min_score: int = 900, hard_ratio: float = 0.5):
    # 1. Load Interactions
    if not STRING_FILE.exists():
        print(f"Error: {STRING_FILE} not found. Please run collect_ppi.py first.")
        return

    inte_df = load_string_interactions(STRING_FILE, min_score=min_score)
    
    # 2. Get unique proteins
    unique_proteins = set(inte_df["protein1"]).union(set(inte_df["protein2"]))
    print(f"Found {len(unique_proteins)} unique proteins in positive set.")
    
    # 4. Generate Negatives
    neg_df = generate_negative_samples(inte_df, list(unique_proteins), ratio=1.0, hard_ratio=hard_ratio)
    
    # 5. Labeling
    inte_df["label"] = 1
    neg_df["label"] = 0
    
    # 6. Combine
    full_df = pd.concat([inte_df, neg_df], ignore_index=True)
    
    # 7. Split
    train_df, test_df = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=42)
    
    # 8. Save
    print("Saving processed data...")
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
