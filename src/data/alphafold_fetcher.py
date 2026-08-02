import requests
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import os

class AlphaFoldFetcher:
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            from src.utils.paths import PROCESSED_DATA_DIR
            self.cache_dir = PROCESSED_DATA_DIR / "alphafold_pdb"
        else:
            self.cache_dir = Path(cache_dir)
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = "https://alphafold.ebi.ac.uk/api/prediction/"

    def fetch_pdb_url(self, uniprot_id: str) -> str:
        """
        Queries AlphaFold DB API to get PDB file download URL for a given UniProt ID.
        """
        try:
            resp = requests.get(f"{self.api_url}{uniprot_id}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("pdbUrl", "")
        except Exception as e:
            print(f"Error querying AlphaFold API for {uniprot_id}: {e}")
        return ""

    def download_pdb(self, uniprot_id: str) -> Path:
        """
        Downloads PDB file for a UniProt ID and saves to cache.
        Returns the path to the cached file.
        """
        dest_path = self.cache_dir / f"{uniprot_id}.pdb"
        if dest_path.exists():
            return dest_path
            
        pdb_url = self.fetch_pdb_url(uniprot_id)
        if not pdb_url:
            # Try fallback direct URL construction
            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            
        try:
            resp = requests.get(pdb_url, timeout=15)
            if resp.status_code == 200:
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"Downloaded AlphaFold PDB for {uniprot_id}")
                return dest_path
        except Exception as e:
            print(f"Failed to download PDB for {uniprot_id} from {pdb_url}: {e}")
            
        return None

    def parse_ca_coordinates(self, pdb_path: Path) -> List[Dict[str, Any]]:
        """
        Parses coordinates of all C-alpha (CA) atoms from a PDB file.
        """
        if not pdb_path or not pdb_path.exists():
            return []
            
        coords = []
        with open(pdb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("ATOM "):
                    atom_name = line[12:16].strip()
                    if atom_name == "CA":
                        res_name = line[17:20].strip()
                        chain_id = line[21].strip()
                        res_seq = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append({
                            "res_seq": res_seq,
                            "res_name": res_name,
                            "chain_id": chain_id,
                            "x": x,
                            "y": y,
                            "z": z
                        })
        return coords

    def parse_cb_coordinates(self, pdb_path: Path) -> List[Dict[str, Any]]:
        """
        Parses coordinates of C-beta (CB) atoms for all residues, falling back to C-alpha (CA) for Glycine.
        """
        if not pdb_path or not pdb_path.exists():
            return []
            
        residue_atoms = {}
        with open(pdb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("ATOM "):
                    atom_name = line[12:16].strip()
                    if atom_name in ("CA", "CB"):
                        res_name = line[17:20].strip()
                        chain_id = line[21].strip()
                        res_seq = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        atom_info = {
                            "res_seq": res_seq,
                            "res_name": res_name,
                            "chain_id": chain_id,
                            "atom_name": atom_name,
                            "x": x,
                            "y": y,
                            "z": z
                        }
                        
                        if res_seq not in residue_atoms:
                            residue_atoms[res_seq] = {}
                        residue_atoms[res_seq][atom_name] = atom_info

        coords = []
        for res_seq in sorted(residue_atoms.keys()):
            atoms = residue_atoms[res_seq]
            if "CB" in atoms:
                coords.append(atoms["CB"])
            elif "CA" in atoms:
                coords.append(atoms["CA"])
        return coords

    def calculate_contact_map(self, coords: List[Dict[str, Any]], threshold: float = 8.0) -> List[Tuple[int, int, float]]:
        """
        Calculates contacts between residues based on 3D distance of CA atoms.
        Returns a list of tuples: (res_seq_i, res_seq_j, distance)
        """
        contacts = []
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = coords[i]
                c2 = coords[j]
                
                # Euclidean distance
                dist = np.sqrt(
                    (c1["x"] - c2["x"])**2 +
                    (c1["y"] - c2["y"])**2 +
                    (c1["z"] - c2["z"])**2
                )
                
                if dist <= threshold:
                    contacts.append((c1["res_seq"], c2["res_seq"], float(dist)))
        return contacts
