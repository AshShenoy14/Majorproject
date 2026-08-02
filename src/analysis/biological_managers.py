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
        return pd.DataFrame(columns=["protein_id", "uniprot_id", "localization", "pathways", "families", "domains"])

    def _save_cache(self):
        self.cache_df.to_csv(self.cache_path, index=False)

    def get_bio_metadata(self, protein_ids: List[str], fetch_missing: bool = True) -> pd.DataFrame:
        """
        Fetches localization and pathway info for proteins.
        protein_ids: List of ENSP IDs.
        """
        # Identify IDs that are either truly missing OR have empty new fields
        existing = self.cache_df[self.cache_df["protein_id"].isin(protein_ids)]
        
        # A protein needs fetching if it's not in cache OR if its new fields are empty
        to_fetch_ids = []
        for pid in protein_ids:
            row = existing[existing["protein_id"] == pid]
            if row.empty:
                to_fetch_ids.append(pid)
            else:
                # Re-fetch if family/domain info is missing (migration case)
                if not str(row.iloc[0].get("families", "")).strip() and not str(row.iloc[0].get("domains", "")).strip():
                    to_fetch_ids.append(pid)
                    # Remove from existing so it gets replaced
                    existing = existing[existing["protein_id"] != pid]
        
        if not to_fetch_ids or not fetch_missing:
            return existing

        # Map ENSP -> UniProt
        mapping = self.mapper.ensp_to_uniprot(to_fetch_ids)
        
        # Fallback: Search UniProt for missing mappings (e.g. if TSV is incomplete)
        unmapped = [p for p in to_fetch_ids if p not in mapping]
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
                    "fields": "accession,cc_subcellular_location,cc_pathway,cc_similarity,cc_domain",
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
                        
                        # Extract Pathways, Families, Domains
                        paths, families, domains = [], [], []
                        for comment in res.get("comments", []):
                            ctype = comment.get("commentType")
                            val = ""
                            if "texts" in comment and comment["texts"]:
                                val = comment["texts"][0].get("value", "")
                            elif "note" in comment and isinstance(comment["note"], dict) and "texts" in comment["note"] and comment["note"]["texts"]:
                                val = comment["note"]["texts"][0].get("value", "")
                            
                            if not val:
                                continue
                                
                            if ctype == "PATHWAY":
                                paths.append(val)
                            elif ctype == "SIMILARITY":
                                families.append(val)
                            elif ctype == "DOMAIN":
                                domains.append(val)
                        
                        new_data.append({
                            "protein_id": ensp,
                            "uniprot_id": uni,
                            "localization": "; ".join(locs),
                            "pathways": "; ".join(paths),
                            "families": "; ".join(families),
                            "domains": "; ".join(domains)
                        })
            except Exception as e:
                print(f"Error fetching bio meta for {uni}: {e}")

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.cache_df = pd.concat([self.cache_df, new_df]).drop_duplicates().reset_index(drop=True)
            self._save_cache()

        return self.cache_df[self.cache_df["protein_id"].isin(protein_ids)]

    def check_biological_compatibility(self, p1_id: str, p2_id: str, fetch_missing: bool = True) -> Dict[str, Any]:
        """
        Checks if two proteins have compatible subcellular localizations AND 
        identifies 'Similarity Traps' (e.g. two co-chaperones with similar domains).
        """
        meta = self.get_bio_metadata([p1_id, p2_id], fetch_missing=fetch_missing)
        if len(meta) < 2:
            return {"compatible": True, "score": 0.5, "reason": "Insufficient biological data", "p1_locs": [], "p2_locs": []}
        
        row1 = meta[meta["protein_id"] == p1_id].iloc[0]
        row2 = meta[meta["protein_id"] == p2_id].iloc[0]
        
        # 1. Localization Check
        l1 = set([x.strip().lower() for x in str(row1["localization"]).split(";") if x.strip()])
        l2 = set([x.strip().lower() for x in str(row2["localization"]).split(";") if x.strip()])
        
        loc_score = 0.5
        if l1 and l2:
            intersection = l1.intersection(l2)
            if len(intersection) > 0:
                loc_score = 1.0
                # Nucleolar bias correction
                is_p1_nucleolar = any("nucleolus" in loc for loc in l1)
                is_p2_nucleolar = any("nucleolus" in loc for loc in l2)
                if is_p1_nucleolar != is_p2_nucleolar: loc_score = 0.7
            else:
                loc_score = 0.1
        
        # 2. Similarity Trap Detection (TTC1/DNAJC7 Guard)
        trap_penalty = 0.0
        f1, f2 = str(row1.get("families", "")).lower(), str(row2.get("families", "")).lower()
        d1, d2 = str(row1.get("domains", "")).lower(), str(row2.get("domains", "")).lower()
        
        # Rule: If they share "Chaperone" or "TPR" language but aren't in the same complex
        # Note: This is a heuristical penalty to force the Graph model to be the tie-breaker
        chaperone_keywords = ["chaperone", "tpr", "tetratricopeptide", "heat shock", "dna j", "hsp"]
        is_p1_chap = any(k in f1 or k in d1 for k in chaperone_keywords)
        is_p2_chap = any(k in f2 or k in d2 for k in chaperone_keywords)
        
        reason = "Shared localization found" if loc_score > 0.5 else "Localization mismatch"
        
        if is_p1_chap and is_p2_chap:
            # Trap detected! Both are chaperones/TPR proteins. 
            # We penalize the bio-score to force the ensemble to rely on Graph evidence.
            trap_penalty = 0.4
            loc_score = max(0.2, loc_score - trap_penalty)
            reason = "Potential 'Similarity Trap' detected: both proteins are co-chaperones/TPR proteins. Reducing confidence in absence of strong graph evidence."
            
        return {
            "compatible": loc_score > 0.3,
            "score": loc_score,
            "reason": reason,
            "p1_locs": list(l1),
            "p2_locs": list(l2),
            "trap_penalty": trap_penalty
        }

    def check_localization_compatibility(self, p1_id: str, p2_id: str, fetch_missing: bool = True) -> Dict[str, Any]:
        """Backwards compatibility alias for the new logic"""
        return self.check_biological_compatibility(p1_id, p2_id, fetch_missing)
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
