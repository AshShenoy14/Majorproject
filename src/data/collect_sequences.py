import requests
import time
from typing import List, Dict
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/search"

def get_sequences(uniprot_ids: List[str], batch_size: int = 50) -> Dict[str, str]:
    """
    Fetches protein sequences from UniProt for a given list of UniProt IDs.
    
    Args:
        uniprot_ids: List of UniProt Accession IDs.
        batch_size: Number of IDs to query in one batch.
        
    Returns:
        Dictionary mapping UniProt ID to sequence.
    """
    sequences = {}
    
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i + batch_size]
        query = " OR ".join([f"accession:{uid}" for uid in batch])
        params = {
            "query": query,
            "format": "fasta"
        }
        
        try:
            response = requests.get(UNIPROT_API_URL, params=params)
            response.raise_for_status()
            
            # Simple FASTA parsing
            current_id = None
            current_seq = []
            
            for line in response.text.splitlines():
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    
                    # Extract ID from header: >sp|P12345|...
                    parts = line.split("|")
                    if len(parts) >= 2:
                        current_id = parts[1]
                    else:
                        current_id = line.split()[0][1:] # Fallback
                    
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            
            if current_id:
                sequences[current_id] = "".join(current_seq)
                
            time.sleep(0.5) # Rate limiting
            
        except requests.RequestException as e:
            print(f"Error fetching batch {i}: {e}")
            
    return sequences

if __name__ == "__main__":
    # Example usage
    test_ids = ["P53_HUMAN", "P04637", "P12345"] # P53 is P04637
    seqs = get_sequences(test_ids)
    print(f"Fetched {len(seqs)} sequences.")
    for uid, seq in seqs.items():
        print(f"{uid}: {seq[:20]}...")
