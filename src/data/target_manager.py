import pandas as pd
from typing import List
from chembl_webresource_client.new_client import new_client
from pathlib import Path
import os
from src.utils.paths import PROCESSED_DATA_DIR

class TargetManager:
    def __init__(self, cache_file: str = "targets_cache.csv"):
        self.cache_path = PROCESSED_DATA_DIR / cache_file
        self.targets_df = self._load_cache()
        self.target_client = new_client.target

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path.exists():
            try:
                return pd.read_csv(self.cache_path)
            except Exception:
                print("Warning: Target cache corrupted. Starting fresh.")
                return pd.DataFrame(columns=["uniprot_id", "chembl_id", "target_name", "target_type"])
        return pd.DataFrame(columns=["uniprot_id", "chembl_id", "target_name", "target_type"])

    def _save_cache(self):
        self.targets_df.to_csv(self.cache_path, index=False)

    def get_targets(self, protein_ids: List[str]) -> pd.DataFrame:
        """
        Get drug targets for a list of valid Protein IDs (UniProt).
        Note: ChEMBL usually needs UniProt Accessions. 
        If 'protein_ids' are ENSP, we might not find hits directly unless mapped.
        """
        # Filter existing
        existing = self.targets_df[self.targets_df["uniprot_id"].isin(protein_ids)]
        found_ids = set(existing["uniprot_id"].unique())
        missing_ids = [p for p in protein_ids if p not in found_ids]

        if not missing_ids:
            return existing

        # Fetch missing
        print(f"Fetching targets for {len(missing_ids)} proteins from ChEMBL...")
        new_targets = self._fetch_from_chembl(missing_ids)
        
        if not new_targets.empty:
            self.targets_df = pd.concat([self.targets_df, new_targets]).drop_duplicates().reset_index(drop=True)
            self._save_cache()
            
        return self.targets_df[self.targets_df["uniprot_id"].isin(protein_ids)]

    def _fetch_from_chembl(self, ids: List[str]) -> pd.DataFrame:
        results = []
        chunk_size = 50
        
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i+chunk_size]
            try:
                # ChEMBL filter by accession
                query = self.target_client.filter(target_components__accession__in=chunk).only(
                    'target_chembl_id', 'pref_name', 'target_type', 'target_components'
                )
                
                for target in query:
                    t_id = target['target_chembl_id']
                    t_name = target['pref_name']
                    t_type = target['target_type']
                    
                    for comp in target['target_components']:
                        if comp['accession'] in chunk:
                            results.append({
                                'uniprot_id': comp['accession'],
                                'chembl_id': t_id,
                                'target_name': t_name,
                                'target_type': t_type
                            })
                            
            except Exception as e:
                print(f"Error fetching Chembl chunk: {e}")
                
        return pd.DataFrame(results)
