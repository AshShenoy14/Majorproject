import os
import sys
import json
import csv
import glob
import urllib.request
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_extraction import ESMFeatureExtractor

# Standard 3-letter to 1-letter amino acid code mapping
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M'
}

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def fetch_pdb_file(pdb_id: str, save_dir: Path) -> Path:
    """Downloads PDB file from RCSB PDB if not locally cached."""
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = save_dir / f"{pdb_id.upper()}.pdb"
    if pdb_path.exists():
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            content = resp.read().decode('utf-8')
            if "NOT FOUND" in content or "404 Not Found" in content:
                raise ValueError(f"PDB {pdb_id} not found on RCSB")
            with open(pdb_path, 'w', encoding='utf-8') as f:
                f.write(content)
        return pdb_path
    except Exception as e:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: {e}")


def parse_all_chains_ca(pdb_path: Path):
    """
    Parses all chains in a PDB file to extract CA coordinates and sequences.
    Returns dict: chain_id -> {'seq': str, 'coords': np.ndarray, 'resnums': list}
    """
    chains_data = {}

    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                current_chain = line[21]
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue

                alt_loc = line[16]
                if alt_loc not in (' ', 'A', '1'):
                    continue

                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                i_code = line[26].strip()
                res_key = (resnum, i_code)

                if resname not in AA_3TO1:
                    continue

                aa_1letter = AA_3TO1[resname]
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue

                if current_chain not in chains_data:
                    chains_data[current_chain] = {
                        'seq_chars': [],
                        'coords': [],
                        'resnums': [],
                        'seen_residues': set()
                    }

                cdata = chains_data[current_chain]
                if res_key in cdata['seen_residues']:
                    continue
                cdata['seen_residues'].add(res_key)

                cdata['seq_chars'].append(aa_1letter)
                cdata['coords'].append([x, y, z])
                cdata['resnums'].append(resnum)

    result = {}
    for ch, cdata in chains_data.items():
        seq = "".join(cdata['seq_chars'])
        if len(seq) >= 10 and set(seq).issubset(VALID_AMINO_ACIDS):
            result[ch] = {
                'seq': seq,
                'coords': np.array(cdata['coords'], dtype=np.float32),
                'resnums': cdata['resnums']
            }

    return result


def find_best_interacting_pair(chains_data: dict, distance_threshold: float = 8.0):
    """
    Finds the pair of distinct chains with the highest number of inter-chain contacts (C_alpha dist <= threshold).
    Returns (chain_a, chain_b, num_contacts, contact_map, mask_a, mask_b) or None.
    """
    chain_ids = sorted(chains_data.keys())
    best_pair = None
    max_contacts = 0
    best_map = None
    best_masks = None

    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            ca, cb = chain_ids[i], chain_ids[j]
            coords_a = chains_data[ca]['coords']
            coords_b = chains_data[cb]['coords']

            diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :] # [L_A, L_B, 3]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))

            cmap = (dists <= distance_threshold).astype(np.uint8)
            num_contacts = int(cmap.sum())

            if num_contacts > max_contacts:
                max_contacts = num_contacts
                mask_a = (cmap.sum(axis=1) > 0).astype(np.uint8)
                mask_b = (cmap.sum(axis=0) > 0).astype(np.uint8)
                best_pair = (ca, cb)
                best_map = cmap
                best_masks = (mask_a, mask_b)

    if best_pair is None or max_contacts == 0:
        return None

    return best_pair[0], best_pair[1], max_contacts, best_map, best_masks[0], best_masks[1]


# Curated benchmark set of experimentally solved PPI complexes from RCSB PDB
BENCHMARK_PDB_IDS = [
    "1YCR", "1BRS", "1DFJ", "1C4Z", "2OOB", "1ACB", "1KAC", "1PPE", "2PTC", "1CHO",
    "1AVW", "1EAW", "1F47", "1GL1", "1HE8", "1JTG", "1MAH", "1N8Z", "1OPS", "1PXV",
    "1QA9", "1R0R", "1T6B", "1UDI", "1VFB", "1WQ1", "1X1U", "1Z0K", "2A1A", "2B42",
    "2COL", "2E10", "2HLE", "2I9B", "2J0T", "2PCC", "2SIC", "2TGP", "2VDB", "3SGB",
    "4CPA", "4H44", "1A2K", "1AK4", "1AY7", "1B39", "1B6C", "1BUH", "1BVN", "1CGI",
    "1D6R", "1E96", "1EO8", "1EZU", "1F34", "1FC2", "1FFW", "1FQJ", "1FSK", "1G4Y",
    "1GCQ", "1GHQ", "1GRN", "1H9D", "1I2M", "1IB1", "1JPS", "1KKL", "1KLU", "1LFD",
    "1M10", "1NSN", "1NVU", "1OFU", "1PXR", "1QFW", "1R69", "1RLB", "1S1Q", "1T8K"
]


def main():
    print("=" * 70)
    print("      BUILDING PILOT IRLM STRUCTURAL DATASET (~50-100 COMPLEXES)")
    print("=" * 70)

    raw_pdb_dir = PROJECT_ROOT / "data" / "raw" / "pdb_structures"
    out_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing ESMFeatureExtractor on device: {device}...")
    extractor = ESMFeatureExtractor(model_name="facebook/esm2_t12_35M_UR50D", device=device)

    manifest_rows = []
    processed_count = 0
    rejected_count = 0
    rejection_reasons = {}

    seen_complex_ids = set()

    for idx, pdb_id in enumerate(BENCHMARK_PDB_IDS, 1):
        print(f"\n[{idx}/{len(BENCHMARK_PDB_IDS)}] Processing PDB ID: {pdb_id}")

        # 1. Download/Load PDB
        try:
            pdb_path = fetch_pdb_file(pdb_id, raw_pdb_dir)
        except Exception as e:
            reason = f"REJECTED_DOWNLOAD_FAILED ({e})"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            manifest_rows.append({
                "complex_id": pdb_id, "pdb_id": pdb_id, "chain_a": "", "chain_b": "",
                "length_a": 0, "length_b": 0, "num_contacts": 0,
                "interface_residues_a": 0, "interface_residues_b": 0,
                "contact_density": 0.0, "status": reason
            })
            continue

        # 2. Parse Chains
        chains_data = parse_all_chains_ca(pdb_path)
        if len(chains_data) < 2:
            reason = f"REJECTED_LESS_THAN_2_CHAINS (found {len(chains_data)} valid chains)"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            manifest_rows.append({
                "complex_id": pdb_id, "pdb_id": pdb_id, "chain_a": "", "chain_b": "",
                "length_a": 0, "length_b": 0, "num_contacts": 0,
                "interface_residues_a": 0, "interface_residues_b": 0,
                "contact_density": 0.0, "status": reason
            })
            continue

        # 3. Find Best Interacting Chain Pair
        pair_info = find_best_interacting_pair(chains_data, distance_threshold=8.0)
        if pair_info is None:
            reason = "REJECTED_ZERO_INTERCHAIN_CONTACTS"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            manifest_rows.append({
                "complex_id": pdb_id, "pdb_id": pdb_id, "chain_a": "", "chain_b": "",
                "length_a": 0, "length_b": 0, "num_contacts": 0,
                "interface_residues_a": 0, "interface_residues_b": 0,
                "contact_density": 0.0, "status": reason
            })
            continue

        chain_a, chain_b, num_contacts, contact_map, mask_a, mask_b = pair_info
        seq_a = chains_data[chain_a]['seq']
        seq_b = chains_data[chain_b]['seq']
        L_A, L_B = len(seq_a), len(seq_b)

        complex_key = f"{pdb_id}_{chain_a}_{chain_b}"
        if complex_key in seen_complex_ids:
            reason = "REJECTED_DUPLICATE_COMPLEX"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            continue
        seen_complex_ids.add(complex_key)

        # Sequence Length Filter (keep sequences manageable for ESM memory)
        if L_A > 600 or L_B > 600:
            reason = f"REJECTED_EXCESSIVE_SEQUENCE_LENGTH (L_A={L_A}, L_B={L_B})"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            manifest_rows.append({
                "complex_id": complex_key, "pdb_id": pdb_id, "chain_a": chain_a, "chain_b": chain_b,
                "length_a": L_A, "length_b": L_B, "num_contacts": num_contacts,
                "interface_residues_a": int(mask_a.sum()), "interface_residues_b": int(mask_b.sum()),
                "contact_density": num_contacts / (L_A * L_B), "status": reason
            })
            continue

        # 4. Generate ESM-2 Residue Embeddings
        try:
            emb_a_tensor = extractor.get_residue_embeddings(seq_a)
            emb_b_tensor = extractor.get_residue_embeddings(seq_b)
            emb_a = emb_a_tensor.cpu().numpy().astype(np.float32)
            emb_b = emb_b_tensor.cpu().numpy().astype(np.float32)
        except Exception as e:
            reason = f"REJECTED_ESM_EMBEDDING_FAILED ({e})"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            manifest_rows.append({
                "complex_id": complex_key, "pdb_id": pdb_id, "chain_a": chain_a, "chain_b": chain_b,
                "length_a": L_A, "length_b": L_B, "num_contacts": num_contacts,
                "interface_residues_a": int(mask_a.sum()), "interface_residues_b": int(mask_b.sum()),
                "contact_density": num_contacts / (L_A * L_B), "status": reason
            })
            continue

        # 5. Strict Validation Checks
        valid_aa = set(seq_a).issubset(VALID_AMINO_ACIDS) and set(seq_b).issubset(VALID_AMINO_ACIDS)
        len_match = (L_A == emb_a.shape[0]) and (L_B == emb_b.shape[0])
        dim_480 = (emb_a.shape[1] == 480) and (emb_b.shape[1] == 480)
        shape_match = (contact_map.shape == (L_A, L_B))
        no_nan_inf = not (np.isnan(emb_a).any() or np.isinf(emb_a).any() or np.isnan(emb_b).any() or np.isinf(emb_b).any())
        binary_map = set(np.unique(contact_map)).issubset({0, 1})

        if not (valid_aa and len_match and dim_480 and shape_match and no_nan_inf and binary_map):
            reason = "REJECTED_VALIDATION_CHECK_FAILED"
            print(f"  --> {reason}")
            rejection_reasons[pdb_id] = reason
            rejected_count += 1
            manifest_rows.append({
                "complex_id": complex_key, "pdb_id": pdb_id, "chain_a": chain_a, "chain_b": chain_b,
                "length_a": L_A, "length_b": L_B, "num_contacts": num_contacts,
                "interface_residues_a": int(mask_a.sum()), "interface_residues_b": int(mask_b.sum()),
                "contact_density": num_contacts / (L_A * L_B), "status": reason
            })
            continue

        # 6. Save NPZ artifact
        npz_filename = out_dir / f"{complex_key}.npz"
        np.savez_compressed(
            npz_filename,
            complex_id=complex_key,
            seq_a=seq_a,
            seq_b=seq_b,
            length_a=L_A,
            length_b=L_B,
            esm_embedding_a=emb_a,
            esm_embedding_b=emb_b,
            contact_map=contact_map,
            interface_mask_a=mask_a,
            interface_mask_b=mask_b
        )

        contact_density = num_contacts / (L_A * L_B)
        processed_count += 1
        print(f"  --> SUCCESS: Saved {npz_filename.name} | L_A={L_A}, L_B={L_B}, contacts={num_contacts}, density={contact_density:.4f}")

        manifest_rows.append({
            "complex_id": complex_key,
            "pdb_id": pdb_id,
            "chain_a": chain_a,
            "chain_b": chain_b,
            "length_a": L_A,
            "length_b": L_B,
            "num_contacts": num_contacts,
            "interface_residues_a": int(mask_a.sum()),
            "interface_residues_b": int(mask_b.sum()),
            "contact_density": round(contact_density, 6),
            "status": "SUCCESS"
        })

    # Save manifest.csv
    fieldnames = [
        "complex_id", "pdb_id", "chain_a", "chain_b", "length_a", "length_b",
        "num_contacts", "interface_residues_a", "interface_residues_b",
        "contact_density", "status"
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 70)
    print("                IRLM DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total Complex Candidates Processed : {len(BENCHMARK_PDB_IDS)}")
    print(f"Successfully Created Artifacts    : {processed_count}")
    print(f"Rejected Candidates                : {rejected_count}")
    print(f"Manifest Location                  : {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
