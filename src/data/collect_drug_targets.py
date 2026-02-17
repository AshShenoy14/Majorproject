from chembl_webresource_client.new_client import new_client
import pandas as pd
from typing import List
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def get_drug_targets(uniprot_ids: List[str]) -> pd.DataFrame:
    """
    Fetches drug target information from ChEMBL for a list of UniProt IDs.
    
    Args:
        uniprot_ids: List of UniProt Accession IDs.
        
    Returns:
        DataFrame containing target information (ChEMBL ID, Target Type, etc.)
    """
    targets = new_client.target
    
    # Filter targets by UniProt Accession
    # ChEMBL API allows filtering by target_components.accession
    
    results = []
    
    # Process in chunks to avoid huge queries
    chunk_size = 50
    for i in range(0, len(uniprot_ids), chunk_size):
        chunk = uniprot_ids[i:i+chunk_size]
        try:
            query = targets.filter(target_components__accession__in=chunk).only(
                'target_chembl_id', 'pref_name', 'target_type', 'target_components'
            )
            
            for target in query:
                # Extract relevant info
                t_id = target['target_chembl_id']
                t_name = target['pref_name']
                t_type = target['target_type']
                
                # Check which specific accession matched (a target can be a complex)
                for comp in target['target_components']:
                     if comp['accession'] in chunk:
                         results.append({
                             'uniprot_id': comp['accession'],
                             'chembl_id': t_id,
                             'target_name': t_name,
                             'target_type': t_type
                         })
                         
        except Exception as e:
            print(f"Error fetching targets for chunk {i}: {e}")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    test_ids = ["P04637"] # p53
    df = get_drug_targets(test_ids)
    print(df.head())
