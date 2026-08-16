import os
import json
import requests
import time
from typing import List, Dict
from pathlib import Path

# Add project root to path if needed (though usually handled by script execution context)
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gzip
from Bio import SeqIO
from src.utils.paths import PROCESSED_DATA_DIR, STRING_SEQUENCES_FILE

class SequenceManager:
    def __init__(self, cache_file: str = "sequences_cache.json"):
        self.cache_path = PROCESSED_DATA_DIR / cache_file
        self.sequences = self._load_cache()
        self.uniprot_api_url = "https://rest.uniprot.org/uniprotkb/search"

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Cache file {self.cache_path} corrupted. Starting fresh.")
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.sequences, f, indent=2)

    def get_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """
        Retrieves sequences for a list of protein IDs.
        Checks cache first, then fetches missing from UniProt.
        Returns a dictionary {protein_id: sequence}.
        """
        results = {}
        missing_ids = []

        # 1. Check Cache
        for pid in protein_ids:
            if pid in self.sequences:
                results[pid] = self.sequences[pid]
            else:
                missing_ids.append(pid)

        if not missing_ids:
            return results

        # 1.5 Try Local File
        local_seqs = self._load_from_local_file(missing_ids)
        if local_seqs:
             print(f"Found {len(local_seqs)} sequences in local file.")
             self.sequences.update(local_seqs)
             self._save_cache()
             results.update(local_seqs)
             
             # Recalculate missing
             missing_ids = [pid for pid in missing_ids if pid not in results]
             
        if not missing_ids:
            return results

        # 2. Fetch Missing
        print(f"Fetching {len(missing_ids)} missing sequences from UniProt...")
        fetched_seqs = self._fetch_from_uniprot(missing_ids)

        # 3. Update Cache & Results
        if fetched_seqs:
            self.sequences.update(fetched_seqs)
            self._save_cache()
            results.update(fetched_seqs)
        
        return results

    def _load_from_local_file(self, missing_ids: List[str]) -> Dict[str, str]:
        found = {}
        if not STRING_SEQUENCES_FILE.exists():
            return found
            
        print(f"Searching local file {STRING_SEQUENCES_FILE} for {len(missing_ids)} sequences...")
        missing_set = set(missing_ids)
        
        try:
            with gzip.open(STRING_SEQUENCES_FILE, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    # ID format: 9606.ENSP...
                    full_id = record.id
                    if full_id.startswith("9606."):
                        pid = full_id[5:]
                    else:
                        pid = full_id
                        
                    if pid in missing_set:
                        found[pid] = str(record.seq)
                        missing_set.remove(pid)
                        if not missing_set:
                            break
        except Exception as e:
            print(f"Error reading local sequences file: {e}")
            
        return found

    def _fetch_direct_uniprot(self, accession: str) -> str:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            lines = r.text.splitlines()
            return "".join(lines[1:]) if len(lines) > 1 else None
        except:
            return None

    def _fetch_from_uniprot(self, ids: List[str], batch_size: int = 50) -> Dict[str, str]:
        fetched = {}
        
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i+batch_size]
            # Construct query: (accession:ID1 OR accession:ID2 OR ...)
            # For ENSP IDs, we might need to search 'ensembl:ENSP...' or just try the ID as query
            # UniProt search supports "ensembl:ENSP00000..."
            
            # Let's try a generic query that matches IDs
            query_parts = []
            for uid in batch:
                if uid.startswith("ENSP"):
                     # Ensembl ID
                     query_parts.append(f"ensembl:{uid}")
                else:
                    # Assume Accession or ID
                    query_parts.append(f"accession:{uid}")
            
            query = " OR ".join(query_parts)
            
            params = {
                "query": query,
                "format": "fasta",
                "size": batch_size
            }

            try:
                response = requests.get(self.uniprot_api_url, params=params)
                response.raise_for_status()
                
                # Parse FASTA
                current_header = None
                current_seq = []
                
                for line in response.text.splitlines():
                    if line.startswith(">"):
                        if current_header:
                            self._map_header_to_id(current_header, "".join(current_seq), batch, fetched)
                        current_header = line
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                
                # Last one
                if current_header:
                    self._map_header_to_id(current_header, "".join(current_seq), batch, fetched)
                    
                time.sleep(0.5) # Rate limit

            except Exception as e:
                print(f"Error fetching batch {i}: {e}")

        return fetched

    def _map_header_to_id(self, header: str, sequence: str, requested_batch: List[str], result_dict: Dict[str, str]):
        """
        Tries to map the FASTA header back to the requested ID.
        Header fmt: >sp|P12345|... or >tr|...
        """
        # This is tricky because UniProt returns Accessions (P12345), but we might have asked for ENSP.
        # However, we need to map it back to the ENSP ID we requested to store it correctly.
        # Simple heuristic: If we asked for list X, and this sequence belongs to one of them...
        # But we don't get the ENSP ID in the FASTA header usually.
        # Problem: We queried using OR. We get a bag of results. We don't know which ENSP maps to which P-ID easily from FASTA.
        
        # BETTER APPROACH for ENSP:
        # If we have ENSP IDs, we should probably map them carefully or use a mapping service first.
        # But for 'search', maybe we can just sequence?
        # Actually, for the purpose of this project, if we can't map back exactly, we might lose the connection.
        
        # Alternative: Query one by one? Too slow.
        # Alternative: Use UniProt ID Mapping API. 
        # But let's try to see if the response includes the query term.
        # The FASTA format doesn't include the Ensembl ID usually.
        
        # Hack for this 'Execution' phase to ensure progress:
        # If we have ENSP IDs, finding the sequence locally or via a specific ENSP-lookup is better.
        # Use Ensembl REST API for ENSP IDs? 
        # Ensembl API: https://rest.ensembl.org/sequence/id/ENSP00000...?content-type=text/x-fasta
        
        # Let's switch strategy: Separation of concerns.
        # If ID starts with ENSP, use Ensembl API.
        # If ID is normal, use UniProt.
        pass

    def _fetch_from_ensembl(self, ensp_id: str) -> str:
        url = f"https://rest.ensembl.org/sequence/id/{ensp_id}"
        params = {"content-type": "text/x-fasta", "type": "protein"}
        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            # Remove header
            lines = r.text.splitlines()
            return "".join(lines[1:])
        except:
            return None

    # Redefine _fetch_from_uniprot to be smarter or use Ensembl for ENSP
    # Overwriting the method below with the hybrid approach
    
    ENSP_FALLBACK_MAP = {
        'ENSP00000269305': 'P04637', # TP53
        'ENSP00000258149': 'Q00987', # MDM2
        'ENSP00000300161': 'O95782', # AP2A2
        'ENSP00000267029': 'Q00610', # CLTC
        'ENSP00000293879': 'Q07812', # BAX
        'ENSP00000307677': 'Q07817', # BCL2L1
        'ENSP00000385802': 'P24941', # CDK2
        'ENSP00000361000': 'P20248', # CCNA2
        'ENSP00000327694': 'P01112', # HRAS
        'ENSP00000373627': 'P01116', # KRAS
        'ENSP00000288135': 'P04629', # NTRK1
        'ENSP00000263967': 'P00533', # EGFR
    }

    def _fetch_from_uniprot_by_ensp(self, ensp_id: str) -> str:
        # Try mapped UniProt accession first
        accession = self.ENSP_FALLBACK_MAP.get(ensp_id)
        if accession:
            seq = self._fetch_direct_uniprot(accession)
            if seq: return seq

        # Try UniProt search by ensembl cross-reference
        url = f"https://rest.uniprot.org/uniprotkb/search?query=ensembl:{ensp_id}&format=fasta&size=1"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            lines = r.text.splitlines()
            if len(lines) > 1 and lines[0].startswith(">"):
                return "".join(lines[1:])
        except Exception:
            pass

        # Try general UniProt query
        url = f"https://rest.uniprot.org/uniprotkb/search?query={ensp_id}&format=fasta&size=1"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            lines = r.text.splitlines()
            if len(lines) > 1 and lines[0].startswith(">"):
                return "".join(lines[1:])
        except Exception:
            pass

        return None

    def _fetch_smart(self, ids: List[str]) -> Dict[str, str]:
        fetched = {}
        
        # Separate ENSP and others
        ensp_ids = [i for i in ids if i.startswith("ENSP")]
        other_ids = [i for i in ids if not i.startswith("ENSP")]
        
        # 1. Ensembl Fetch (with UniProt fallback)
        if ensp_ids:
            print(f"Fetching {len(ensp_ids)} IDs from Ensembl...")
            for i, eid in enumerate(ensp_ids):
                seq = None
                try:
                    seq = self._fetch_from_ensembl(eid)
                except Exception as e:
                    print(f"Ensembl fetch error for {eid}: {e}")
                
                if not seq:
                    print(f"Ensembl returned no sequence for {eid}, trying UniProt fallback...")
                    seq = self._fetch_from_uniprot_by_ensp(eid)

                if seq:
                    fetched[eid] = seq
                    print(f"Successfully retrieved sequence for {eid}")
                else:
                    print(f"Could not retrieve sequence for {eid} from Ensembl or UniProt")
                
                time.sleep(0.1) 
        
        # 2. UniProt Fetch 
        if other_ids:
             print(f"Fetching {len(other_ids)} IDs from UniProt...")
             for uid in other_ids:
                 seq = self._fetch_direct_uniprot(uid)
                 if seq:
                     fetched[uid] = seq
                 else:
                     print(f"UniProt returned no sequence for {uid}")
             
        return fetched

    # Replace the main fetch caller with _fetch_smart logic
    def _fetch_from_uniprot(self, ids: List[str]) -> Dict[str, str]:
        return self._fetch_smart(ids)

