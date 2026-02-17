import requests
import gzip
import shutil
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import RAW_DATA_DIR, STRING_FILE

def download_string_db():
    """Downloads human protein-protein interactions from STRING DB."""
    url = "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    
    if STRING_FILE.exists():
        print(f"STRING file already exists at {STRING_FILE}")
        return

    print(f"Downloading STRING DB from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(STRING_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Interactions download complete.")
    except Exception as e:
        print(f"Failed to download STRING DB interactions: {e}")

    # Download Sequences
    seq_url = "https://stringdb-static.org/download/protein.sequences.v12.0/9606.protein.sequences.v12.0.fa.gz"
    from src.utils.paths import STRING_SEQUENCES_FILE
    
    if STRING_SEQUENCES_FILE.exists():
        print(f"STRING sequences file already exists at {STRING_SEQUENCES_FILE}")
        return

    print(f"Downloading STRING Sequences from {seq_url}...")
    try:
        response = requests.get(seq_url, stream=True)
        response.raise_for_status()
        
        with open(STRING_SEQUENCES_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Sequences download complete.")
    except Exception as e:
        print(f"Failed to download STRING DB sequences: {e}")

def download_biogrid():
    """Placeholder for BioGRID download. BioGRID requires a specific release URL."""
    # BioGRID URLs change with versions, so we might need a dynamic way or fixed version.
    # For now, we will just print a message as it often requires login or specific version tracking.
    print("Please download the latest BioGRID release for Homo sapiens manually if not present.")
    print("Place it in: ", RAW_DATA_DIR)

if __name__ == "__main__":
    download_string_db()
    download_biogrid()
