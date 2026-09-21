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

def _canonical(p1: pd.Series, p2: pd.Series):
    a = np.where(p1 < p2, p1, p2)
    b = np.where(p1 < p2, p2, p1)
    return a, b


class PairSet:
    """Memory-compact set of canonical protein pairs (each pair packed into one int64)."""

    def __init__(self, proteins: List[str], first: np.ndarray, second: np.ndarray):
        self.index = {p: i for i, p in enumerate(proteins)}
        self.n = len(proteins)
        idx = np.vectorize(self.index.__getitem__, otypes=[np.int64])
        self._keys = set((idx(first) * self.n + idx(second)).tolist()) if len(first) else set()

    def __len__(self):
        return len(self._keys)

    def contains(self, u: str, v: str) -> bool:
        """u, v must be canonical (u < v). Proteins unknown to STRING cannot be members."""
        iu, iv = self.index.get(u), self.index.get(v)
        return iu is not None and iv is not None and (iu * self.n + iv) in self._keys

    def intersect_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows of df (canonicalised on the fly) that are members of the set."""
        if df.empty:
            return df
        a, b = _canonical(df["protein1"], df["protein2"])
        mask = np.fromiter((self.contains(x, y) for x, y in zip(a, b)), dtype=bool, count=len(df))
        return df[mask]


def load_string_interactions(filepath: str, min_score: int = 900) -> Tuple[pd.DataFrame, PairSet]:
    """
    Loads the STRING file ONCE and returns two views of it:
      positive_df     - canonical unique pairs with combined_score >= min_score (the positive class)
      exclusion_pairs - canonical pairs with combined_score > 0, i.e. ANY confidence level.
    A negative candidate must not be in exclusion_pairs: a pair STRING scores at 350 is still a
    reported interaction, so labelling it 0 just because it is below min_score contaminates the negatives.
    STRING IDs are '9606.ENSP...'; the taxon prefix is stripped.
    """
    print(f"Loading interactions from {filepath}...")
    df = pd.read_csv(filepath, sep=" ", usecols=["protein1", "protein2", "combined_score"])

    df["protein1"] = df["protein1"].astype(str).str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].astype(str).str.replace("9606.", "", regex=False)
    df = df[df["protein1"] != df["protein2"]]

    a, b = _canonical(df["protein1"], df["protein2"])
    canon = pd.DataFrame({"protein1": a, "protein2": b, "combined_score": df["combined_score"].values})
    canon = canon.drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)

    positive_df = canon[canon["combined_score"] >= min_score][["protein1", "protein2"]].reset_index(drop=True)
    any_conf = canon[canon["combined_score"] > 0]
    proteins = sorted(set(canon["protein1"]).union(canon["protein2"]))
    exclusion_pairs = PairSet(proteins, any_conf["protein1"].values, any_conf["protein2"].values)

    print(f"Loaded {len(positive_df)} canonical positive interactions (score >= {min_score}).")
    print(f"Exclusion set for negatives: {len(exclusion_pairs)} canonical pairs with score > 0.")
    return positive_df, exclusion_pairs

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

def generate_hard_negatives(positive_df: pd.DataFrame, all_proteins: List[str], exclusion_pairs: PairSet, ratio: float = 0.5, loc_dict: dict = None) -> pd.DataFrame:
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
    
    hard_negatives = set()
    num_needed = int(len(positive_df) * ratio)

    # sorted() before shuffling: set/dict iteration order over strings varies with PYTHONHASHSEED,
    # which would make the sample differ between runs even with a fixed numpy seed.
    p_list = sorted(adj.keys())
    np.random.shuffle(p_list)
    
    for u in p_list:
        if len(hard_negatives) >= num_needed:
            break
        
        for neighbor in sorted(adj[u]):
            for v in sorted(adj[neighbor]):
                if u == v:
                    continue
                
                u_c, v_c = (u, v) if u < v else (v, u)
                
                if not exclusion_pairs.contains(u_c, v_c) and (u_c, v_c) not in hard_negatives:
                    if loc_dict is None or is_co_localized(u_c, v_c, loc_dict):
                        hard_negatives.add((u_c, v_c))
                        if len(hard_negatives) >= num_needed:
                            break
            if len(hard_negatives) >= num_needed:
                break
            
    print(f"Generated {len(hard_negatives)} hard negatives.")
    if hard_negatives:
        return pd.DataFrame(sorted(hard_negatives), columns=["protein1", "protein2"])
    else:
        return pd.DataFrame(columns=["protein1", "protein2"])

def generate_negative_samples(positive_df: pd.DataFrame, all_proteins: List[str], exclusion_pairs: PairSet, ratio: float = 1.0, hard_ratio: float = 0.5, loc_dict: dict = None) -> pd.DataFrame:
    """
    Generates negative samples by mixing random (easy) and common-neighbor (hard) pairs.
    All pairs are canonicalized (protein1 < protein2) and strictly deduplicated.
    """
    total_needed = int(len(positive_df) * ratio)
    
    hard_df = generate_hard_negatives(positive_df, all_proteins, exclusion_pairs, ratio=hard_ratio, loc_dict=loc_dict)
    num_hard = len(hard_df)
    num_easy = total_needed - num_hard
    
    print(f"Generating {num_easy} easy (random) negative samples with localization constraints...")
    hard_pairs = set(zip(hard_df["protein1"], hard_df["protein2"])) if not hard_df.empty else set()
    negative_pairs = set()
    
    all_proteins_arr = np.array(all_proteins)
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
            
            if not exclusion_pairs.contains(u_c, v_c) and (u_c, v_c) not in hard_pairs and (u_c, v_c) not in negative_pairs:
                if loc_dict is None or is_co_localized(u_c, v_c, loc_dict):
                    negative_pairs.add((u_c, v_c))
                    if len(negative_pairs) >= num_easy:
                        break
                        
    easy_df = pd.DataFrame(sorted(negative_pairs), columns=["protein1", "protein2"])
    full_neg_df = pd.concat([easy_df, hard_df], ignore_index=True).drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)
    print(f"Total unique negative samples generated: {len(full_neg_df)}")
    return full_neg_df

def preprocess_data(min_score: int = 900, hard_ratio: float = 0.5, seed: int = 42):
    # 1. Load interactions ONCE: positives (score >= min_score) and the any-confidence exclusion set
    if not STRING_FILE.exists():
        print(f"Error: {STRING_FILE} not found. Please run collect_ppi.py first.")
        return

    inte_df, exclusion_pairs = load_string_interactions(STRING_FILE, min_score=min_score)

    # 2. Get unique proteins
    unique_proteins = sorted(list(set(inte_df["protein1"]).union(set(inte_df["protein2"]))))
    print(f"Found {len(unique_proteins)} unique proteins in positive set.")

    # 3. Load localization cache
    loc_dict = load_localization_cache()

    # 4. Generate Negatives (every candidate is checked against the any-confidence exclusion set)
    neg_df = generate_negative_samples(inte_df, unique_proteins, exclusion_pairs, ratio=1.0, hard_ratio=hard_ratio, loc_dict=loc_dict)

    # 5. Labeling
    inte_df["label"] = 1
    neg_df["label"] = 0

    # 6. Combine & Final Deduplication Check
    full_df = pd.concat([inte_df, neg_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)

    print(f"Full dataset size: {len(full_df)} (Positives: {(full_df['label'] == 1).sum()}, Negatives: {(full_df['label'] == 0).sum()})")

    # 7. Leak-Free Pair Splitting
    train_df, test_temp_df = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=seed)
    val_df, test_df = train_test_split(test_temp_df, test_size=0.5, stratify=test_temp_df["label"], random_state=seed)

    # 8. Hard contamination check on the FINAL splits, before anything is written to disk
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        contaminated = exclusion_pairs.intersect_df(df[df["label"] == 0])
        if len(contaminated) > 0:
            raise RuntimeError(
                f"Negative contamination in {name}: {len(contaminated)} negatives are STRING interactions "
                f"(score > 0). Example: {contaminated.iloc[0][['protein1', 'protein2']].tolist()}"
            )
    print("Negative contamination check passed: 0 negatives in train/val/test appear in the STRING "
          "any-confidence (score > 0) interaction set.")

    # 9. Save
    print("Saving processed clean dataset...")
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    import argparse
    from src.utils.seed import set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_score", type=int, default=900)
    parser.add_argument("--hard_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    preprocess_data(min_score=args.min_score, hard_ratio=args.hard_ratio, seed=args.seed)
