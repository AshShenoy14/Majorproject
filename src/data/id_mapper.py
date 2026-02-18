import pandas as pd
from pathlib import Path
from src.utils.paths import RAW_DATA_DIR

class IDMapper:
    def __init__(self, mapping_file: str = "idmapping_2026_02_16.tsv"):
        self.mapping_path = RAW_DATA_DIR / mapping_file
        self.map_df = self._load_mapping()
        
    def _load_mapping(self):
        if self.mapping_path.exists():
            try:
                # Read TSV
                return pd.read_csv(self.mapping_path, sep='\t')
            except Exception as e:
                print(f"Error loading ID mapping: {e}")
                return pd.DataFrame()
        else:
            print(f"Warning: ID mapping file not found at {self.mapping_path}")
            return pd.DataFrame()

    def ensp_to_uniprot(self, ensp_ids: list) -> dict:
        """
        Returns a dictionary {ensp_id: uniprot_id} for found mappings.
        """
        if self.map_df.empty:
            return {}
            
        # Filter where 'From' (ENSP) is in input list
        # 'From' column likely contains ENSP IDs
        subset = self.map_df[self.map_df['From'].isin(ensp_ids)]
        
        # Create dict
        return dict(zip(subset['From'], subset['Entry']))

    def uniprot_to_ensp(self, uniprot_ids: list) -> dict:
        """
        Returns a dictionary {uniprot_id: ensp_id}
        """
        if self.map_df.empty:
            return {}
            
        subset = self.map_df[self.map_df['Entry'].isin(uniprot_ids)]
        return dict(zip(subset['Entry'], subset['From']))
