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

def generate_negative_samples(positive_df: pd.DataFrame, all_proteins: List[str], ratio: float = 1.0) -> pd.DataFrame:
    """
    Generates negative samples (non-interacting pairs) by random sampling.
    """
    print("Generating negative samples...")
    positive_pairs = set(zip(positive_df["protein1"], positive_df["protein2"]))
    negative_pairs = set()
    
    num_negatives = int(len(positive_df) * ratio)
    all_proteins_arr = np.array(all_proteins)
    
    while len(negative_pairs) < num_negatives:
        # Sample random pairs
        p1 = np.random.choice(all_proteins_arr, num_negatives, replace=True)
        p2 = np.random.choice(all_proteins_arr, num_negatives, replace=True)
        
        for u, v in zip(p1, p2):
            if u == v: continue
            if u > v: u, v = v, u # canonical order
            
            if (u, v) not in positive_pairs and (u, v) not in negative_pairs:
                negative_pairs.add((u, v))
                if len(negative_pairs) >= num_negatives:
                    break
                    
    neg_df = pd.DataFrame(list(negative_pairs), columns=["protein1", "protein2"])
    return neg_df

def preprocess_data(min_score: int = 900):
    # 1. Load Interactions
    if not STRING_FILE.exists():
        print(f"Error: {STRING_FILE} not found. Please run collect_ppi.py first.")
        # Create a dummy file for demonstration if it doesn't exist?
        # No, better to fail and inform.
        return

    inte_df = load_string_interactions(STRING_FILE, min_score=min_score)
    
    # 2. Get unique proteins
    unique_proteins = set(inte_df["protein1"]).union(set(inte_df["protein2"]))
    print(f"Found {len(unique_proteins)} unique proteins in positive set.")
    
    # 3. Load Sequences (Optional for now, but crucial for model)
    # seqs = load_sequences(UNIPROT_FILE)
    # For now, we proceed without checking sequence existence to allow logic verification
    
    # 4. Generate Negatives
    neg_df = generate_negative_samples(inte_df, list(unique_proteins), ratio=1.0)
    
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
    preprocess_data()
