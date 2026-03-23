import requests
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from src.utils.paths import PROCESSED_DATA_DIR
from src.data.id_mapper import IDMapper

class BiologicalManager:
    def __init__(self, cache_file: str = "bio_metadata_cache.csv"):
        self.cache_path = PROCESSED_DATA_DIR / cache_file
        self.cache_df = self._load_cache()
        self.mapper = IDMapper()
        self.uniprot_url = "https://rest.uniprot.org/uniprotkb/search"

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path.exists():
            return pd.read_csv(self.cache_path).fillna("")
        return pd.DataFrame(columns=["protein_id", "uniprot_id", "localization", "pathways"])

    def _save_cache(self):
        self.cache_df.to_csv(self.cache_path, index=False)

    def get_bio_metadata(self, protein_ids: List[str], fetch_missing: bool = True) -> pd.DataFrame:
        """
        Fetches localization and pathway info for proteins.
        protein_ids: List of ENSP IDs.
        """
        existing = self.cache_df[self.cache_df["protein_id"].isin(protein_ids)]
        found_ids = set(existing["protein_id"].unique())
        missing_ids = [p for p in protein_ids if p not in found_ids]

        if not missing_ids or not fetch_missing:
            return existing

        # Map ENSP -> UniProt
        mapping = self.mapper.ensp_to_uniprot(missing_ids)
        
        # Fallback: Search UniProt for missing mappings (e.g. if TSV is incomplete)
        unmapped = [p for p in missing_ids if p not in mapping]
        if unmapped and fetch_missing:
            print(f"Fallback: Searching UniProt for {len(unmapped)} unmapped IDs...")
            for p in unmapped:
                try:
                    search_resp = requests.get(self.uniprot_url, params={"query": p, "format": "json"}, timeout=10)
                    if search_resp.status_code == 200:
                        s_data = search_resp.json()
                        if s_data.get("results"):
                            uni_id = s_data["results"][0].get("primaryAccession")
                            if uni_id:
                                mapping[p] = uni_id
                                print(f"Mapped {p} -> {uni_id} via search.")
                except Exception as e:
                    print(f"Search fallback failed for {p}: {e}")

        if not mapping:
            return existing

        new_data = []
        for ensp, uni in mapping.items():
            try:
                # Fetch from UniProt
                params = {
                    "query": f"accession:{uni}",
                    "fields": "accession,cc_subcellular_location,cc_pathway",
                    "format": "json"
                }
                resp = requests.get(self.uniprot_url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("results"):
                        res = data["results"][0]
                        # Extract Localization
                        locs = []
                        for comment in res.get("comments", []):
                            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                                for loc in comment.get("subcellularLocations", []):
                                    locs.append(loc.get("location", {}).get("value", ""))
                        
                        # Extract Pathways
                        paths = []
                        for comment in res.get("comments", []):
                            if comment.get("commentType") == "PATHWAY":
                                paths.append(comment.get("note", {}).get("texts", [{}])[0].get("value", ""))
                        
                        new_data.append({
                            "protein_id": ensp,
                            "uniprot_id": uni,
                            "localization": "; ".join(locs),
                            "pathways": "; ".join(paths)
                        })
            except Exception as e:
                print(f"Error fetching bio meta for {uni}: {e}")

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.cache_df = pd.concat([self.cache_df, new_df]).drop_duplicates().reset_index(drop=True)
            self._save_cache()

        return self.cache_df[self.cache_df["protein_id"].isin(protein_ids)]

    def check_localization_compatibility(self, p1_id: str, p2_id: str, fetch_missing: bool = True) -> Dict[str, Any]:
        """
        Checks if two proteins have overlapping subcellular localizations and returns a score.
        Score 1.0: Clear overlap
        Score 0.5: Missing data or weak overlap (e.g. Nucleus vs Nucleolus)
        Score 0.1: No overlap
        """
        meta = self.get_bio_metadata([p1_id, p2_id], fetch_missing=fetch_missing)
        if len(meta) < 2:
            return {"compatible": True, "score": 0.5, "reason": "Insufficient localization data", "p1_locs": [], "p2_locs": []}
        
        row1 = meta[meta["protein_id"] == p1_id].iloc[0]
        row2 = meta[meta["protein_id"] == p2_id].iloc[0]
        
        l1 = set([x.strip().lower() for x in str(row1["localization"]).split(";") if x.strip()])
        l2 = set([x.strip().lower() for x in str(row2["localization"]).split(";") if x.strip()])
        
        if not l1 or not l2:
            return {"compatible": True, "score": 0.5, "reason": "Missing localization for one or both proteins", "p1_locs": list(l1), "p2_locs": list(l2)}
        
        intersection = l1.intersection(l2)
        
        # High-level compatibility: Shared location
        if len(intersection) > 0:
            # Granular Check: Nucleus vs Nucleolus
            # If one is ONLY in the nucleolus and the other is in the nucleus but NOT nucleolus
            is_p1_nucleolar = any("nucleolus" in loc for loc in l1)
            is_p2_nucleolar = any("nucleolus" in loc for loc in l2)
            
            if is_p1_nucleolar != is_p2_nucleolar:
                # One is nucleolar, the other isn't. Weakened compatibility.
                return {
                    "compatible": True, 
                    "score": 0.7, 
                    "intersection": list(intersection),
                    "reason": "One protein is nucleolar-specific; partial overlap in nucleus.",
                    "p1_locs": list(l1), "p2_locs": list(l2)
                }
            
            return {
                "compatible": True, 
                "score": 1.0, 
                "intersection": list(intersection),
                "reason": "Strong shared localization found",
                "p1_locs": list(l1), "p2_locs": list(l2)
            }
        
        return {
            "compatible": False,
            "score": 0.1,
            "intersection": [],
            "p1_locs": list(l1),
            "p2_locs": list(l2),
            "reason": "No shared subcellular localization found"
        }
    def calculate_pathway_vulnerability(self, p1_id: str, p2_id: str, delta_score: float) -> Dict[str, Any]:
        """
        Estimates the risk to biological pathways if this interaction is disrupted.
        """
        meta = self.get_bio_metadata([p1_id, p2_id])
        pathways = []
        if not meta.empty:
            for _, row in meta.iterrows():
                if pd.notna(row['pathways']):
                    pathways.extend([p.strip() for p in str(row['pathways']).split(";") if p.strip()])
        
        pathways = list(set(pathways))
        
        # Risk Score Logic: 
        # If the interaction drop (delta_score) is high, the pathways associated 
        # with these proteins are considered "at risk".
        risk_level = "Low"
        if abs(delta_score) > 0.1:
            risk_level = "Moderate"
        if abs(delta_score) > 0.25:
            risk_level = "High"
            
        return {
            "interaction": f"{p1_id}<->{p2_id}",
            "impact_delta": delta_score,
            "risk_level": risk_level,
            "affected_pathways": pathways,
            "description": f"A {risk_level} risk detected for {len(pathways)} cellular pathways due to interaction variance."
        }
