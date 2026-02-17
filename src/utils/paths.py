import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
STRING_FILE = RAW_DATA_DIR / "9606.protein.links.v12.0.txt.gz"
STRING_SEQUENCES_FILE = RAW_DATA_DIR / "9606.protein.sequences.v12.0.fa.gz"
BIOGRID_FILE = RAW_DATA_DIR / "BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.txt"
UNIPROT_FILE = RAW_DATA_DIR / "uniprot_human.fasta"
CHEMBL_FILE = RAW_DATA_DIR / "chembl_targets.csv"
