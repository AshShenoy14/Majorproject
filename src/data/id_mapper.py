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
        Handles both raw ENSP IDs and prefixed IDs (e.g. '9606.ENSP...').
        """
        if self.map_df.empty:
            return {}
            
        id_clean_map = {orig: orig.split(".", 1)[-1] if "." in orig else orig for orig in ensp_ids}
        clean_ids = list(set(id_clean_map.values()))
        
        subset = self.map_df[self.map_df['From'].isin(clean_ids)]
        found_map = dict(zip(subset['From'], subset['Entry']))
        
        result = {}
        for orig, clean in id_clean_map.items():
            if clean in found_map:
                result[orig] = found_map[clean]
            elif orig in found_map:
                result[orig] = found_map[orig]
        return result

    def uniprot_to_ensp(self, uniprot_ids: list) -> dict:
        """
        Returns a dictionary {uniprot_id: ensp_id}
        """
        if self.map_df.empty:
            return {}
            
        subset = self.map_df[self.map_df['Entry'].isin(uniprot_ids)]
        return dict(zip(subset['Entry'], subset['From']))

    def resolve_to_graph_id(self, protein_id: str, graph_ids: set) -> str:
        """
        If protein_id is in graph_ids, returns it.
        Otherwise, looks for alternative IDs (isoforms) of the same protein that ARE in graph_ids.
        """
        if protein_id in graph_ids:
            return protein_id

        if self.map_df.empty:
            return protein_id
            
        # 1. Find the UniProt Entry for this ID
        # 'From' column contains original IDs (usually ENSP)
        entry_row = self.map_df[self.map_df['From'] == protein_id]
        if entry_row.empty:
            return protein_id
            
        uniprot_entry = entry_row.iloc[0]['Entry']
        
        # 2. Find all other ENSP IDs for this UniProt Entry
        all_isoforms = self.map_df[self.map_df['Entry'] == uniprot_entry]['From'].tolist()
        
        # 3. Check if any isoform is in graph_ids
        for iso in all_isoforms:
            if iso in graph_ids:
                print(f"Resilient Mapping: Resolved {protein_id} -> {iso} (Isoform Switch)")
                return iso
                
        return protein_id
