import pandas as pd
from typing import List
from pathlib import Path
import os
from src.utils.paths import PROCESSED_DATA_DIR

from src.data.id_mapper import IDMapper

class TargetManager:
    def __init__(self, cache_file: str = "chembl_targets.csv"):
        self.cache_path = PROCESSED_DATA_DIR / cache_file
        self.targets_df = self._load_cache()
        try:
            from chembl_webresource_client.new_client import new_client
            self.target_client = new_client.target
        except Exception as e:
            print(f"Warning: ChEMBL API is unreachable ({e}). Drug target generation will be limited.")
            self.target_client = None
        self.mapper = IDMapper()

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path.exists():
            try:
                return pd.read_csv(self.cache_path)
            except Exception:
                print("Warning: Target cache corrupted. Starting fresh.")
                return pd.DataFrame(columns=["protein_id", "uniprot_id", "chembl_id", "target_name", "target_type"])
        return pd.DataFrame(columns=["protein_id", "uniprot_id", "chembl_id", "target_name", "target_type"])

    def _save_cache(self):
        self.targets_df.to_csv(self.cache_path, index=False)

    def get_targets(self, protein_ids: List[str]) -> pd.DataFrame:
        """
        Get drug targets for a list of Proteins (ENSP IDs).
        """
        # 1. Check Cache
        # Cache now stores 'protein_id' (ENSP) as primary key equivalent
        if "protein_id" not in self.targets_df.columns:
             # Legacy cache fix if valid
             pass
        
        # Filter existing by protein_id
        existing = self.targets_df[self.targets_df["protein_id"].isin(protein_ids)]
        found_ids = set(existing["protein_id"].unique())
        missing_ids = [p for p in protein_ids if p not in found_ids]

        if not missing_ids:
            return existing

        # 2. Map Missing ENSP -> UniProt
        print(f"Mapping {len(missing_ids)} missing proteins to UniProt...")
        mapping = self.mapper.ensp_to_uniprot(missing_ids)
        
        # If no mapping found for some, we can't fetch them from ChEMBL easily
        valid_missing_map = {ensp: uni for ensp, uni in mapping.items()}
        
        if not valid_missing_map:
            print("No valid UniProt mappings found for missing IDs.")
            return existing

        # 3. Fetch from ChEMBL using UniProt IDs
        print(f"Fetching targets for {len(valid_missing_map)} proteins from ChEMBL...")
        new_targets = self._fetch_from_chembl(valid_missing_map)
        
        if not new_targets.empty:
            self.targets_df = pd.concat([self.targets_df, new_targets]).drop_duplicates().reset_index(drop=True)
            self._save_cache()
            
        return self.targets_df[self.targets_df["protein_id"].isin(protein_ids)]

    def _fetch_from_chembl(self, mapping_dict: dict) -> pd.DataFrame:
        """
        mapping_dict: {ensp_id: uniprot_id}
        """
        if self.target_client is None:
            return pd.DataFrame()
            
        results = []
        uniprot_to_ensp = {v: k for k, v in mapping_dict.items()}
        uniprot_ids = list(mapping_dict.values())
        
        chunk_size = 50
        
        for i in range(0, len(uniprot_ids), chunk_size):
            chunk = uniprot_ids[i:i+chunk_size]
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
                        acc = comp['accession']
                        if acc in chunk and acc in uniprot_to_ensp:
                            results.append({
                                'protein_id': uniprot_to_ensp[acc], # Original ENSP
                                'uniprot_id': acc,
                                'chembl_id': t_id,
                                'target_name': t_name,
                                'target_type': t_type
                            })
                            
            except Exception as e:
                print(f"Error fetching Chembl chunk: {e}")
                
        return pd.DataFrame(results)
