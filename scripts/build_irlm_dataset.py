import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

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

def parse_pdb_chain_ca(pdb_path: Path, chain_id: str):
    """
    Parses a PDB file to extract single-letter amino acid sequence and 
    C-alpha atomic coordinates (x, y, z) for a given chain ID.
    Returns:
        sequence: str
        coords: np.ndarray of shape [L, 3]
        resnums: list of int
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    sequence_chars = []
    coords_list = []
    resnum_list = []

    seen_residues = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                current_chain = line[21]
                if current_chain != chain_id:
                    continue

                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue

                # Alternate location indicator (line[16]) - take primary or 'A'
                alt_loc = line[16]
                if alt_loc not in (' ', 'A', '1'):
                    continue

                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                i_code = line[26].strip()
                res_key = (resnum, i_code)

                # Avoid duplicate CA atoms for same residue number
                if res_key in seen_residues:
                    continue
                seen_residues.add(res_key)

                if resname not in AA_3TO1:
                    print(f"Warning: Non-standard residue '{resname}' at {chain_id}:{resnum} skipped or unmapped.")
                    continue

                aa_1letter = AA_3TO1[resname]
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                sequence_chars.append(aa_1letter)
                coords_list.append([x, y, z])
                resnum_list.append(resnum)

    if not sequence_chars:
        raise ValueError(f"No C-alpha atoms found for Chain '{chain_id}' in {pdb_path}")

    sequence = "".join(sequence_chars)
    coords = np.array(coords_list, dtype=np.float32)

    return sequence, coords, resnum_list

def generate_contact_map(coords_a: np.ndarray, coords_b: np.ndarray, distance_threshold: float = 8.0):
    """
    Computes pairwise Euclidean distances between C-alpha coordinates of Chain A and Chain B.
    Returns:
        contact_map: np.ndarray of shape [L_A, L_B] (uint8 0/1)
        interface_mask_a: np.ndarray of shape [L_A] (uint8 0/1)
        interface_mask_b: np.ndarray of shape [L_B] (uint8 0/1)
    """
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :] # [L_A, L_B, 3]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1)) # [L_A, L_B]

    contact_map = (dists <= distance_threshold).astype(np.uint8)
    interface_mask_a = (contact_map.sum(axis=1) > 0).astype(np.uint8)
    interface_mask_b = (contact_map.sum(axis=0) > 0).astype(np.uint8)

    return contact_map, interface_mask_a, interface_mask_b

def build_irlm_complex(pdb_path: Path, chain_a: str, chain_b: str, complex_id: str, extractor: ESMFeatureExtractor):
    """
    Processes a single PDB complex to generate residue contact maps and ESM-2 embeddings.
    """
    print(f"\nProcessing PDB: {pdb_path.name} | Chain A: {chain_a} | Chain B: {chain_b}")

    # 1. Extract Sequences and CA Coordinates
    seq_a, coords_a, resnums_a = parse_pdb_chain_ca(pdb_path, chain_a)
    seq_b, coords_b, resnums_b = parse_pdb_chain_ca(pdb_path, chain_b)

    L_A = len(seq_a)
    L_B = len(seq_b)

    print(f"Extracted Sequence A (length {L_A}): {seq_a[:30]}...")
    print(f"Extracted Sequence B (length {L_B}): {seq_b[:30]}...")

    # 2. Compute 2D Contact Map and Interface Masks
    contact_map, mask_a, mask_b = generate_contact_map(coords_a, coords_b, distance_threshold=8.0)

    # 3. Extract ESM-2 Residue Embeddings using ESMFeatureExtractor
    print("Generating ESM-2 unpooled residue embeddings...")
    emb_a_tensor = extractor.get_residue_embeddings(seq_a)
    emb_b_tensor = extractor.get_residue_embeddings(seq_b)

    emb_a = emb_a_tensor.cpu().numpy().astype(np.float32)
    emb_b = emb_b_tensor.cpu().numpy().astype(np.float32)

    # 4. Strict Validation Checks
    validation_results = {}

    # Check 1: Valid Amino Acid Sequence
    is_valid_aa_a = set(seq_a).issubset(VALID_AMINO_ACIDS)
    is_valid_aa_b = set(seq_b).issubset(VALID_AMINO_ACIDS)
    validation_results["valid_amino_acids"] = is_valid_aa_a and is_valid_aa_b

    # Check 2: Sequence Length == Embedding Length
    validation_results["length_matches_emb_a"] = (L_A == emb_a.shape[0])
    validation_results["length_matches_emb_b"] = (L_B == emb_b.shape[0])

    # Check 3: Embedding Dimension == 480
    validation_results["emb_dim_a_is_480"] = (emb_a.shape[1] == 480)
    validation_results["emb_dim_b_is_480"] = (emb_b.shape[1] == 480)

    # Check 4: Contact Map Shape == (L_A, L_B)
    validation_results["contact_map_shape_correct"] = (contact_map.shape == (L_A, L_B))

    # Check 5: Interface Mask Lengths
    validation_results["mask_a_length_correct"] = (len(mask_a) == L_A)
    validation_results["mask_b_length_correct"] = (len(mask_b) == L_B)

    # Check 6: No NaN or Inf in Embeddings
    validation_results["no_nan_inf_emb_a"] = not (np.isnan(emb_a).any() or np.isinf(emb_a).any())
    validation_results["no_nan_inf_emb_b"] = not (np.isnan(emb_b).any() or np.isinf(emb_b).any())

    # Check 7: Contact map contains strictly 0 and 1
    unique_contacts = set(np.unique(contact_map))
    validation_results["contact_map_binary_only"] = unique_contacts.issubset({0, 1})

    all_passed = all(validation_results.values())

    # 5. Output File Path
    out_dir = PROJECT_ROOT / "data" / "processed" / "irlm_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{complex_id}_{chain_a}_{chain_b}.npz"

    # Save to NPZ format
    np.savez_compressed(
        out_file,
        complex_id=complex_id,
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

    # 6. Detailed Validation Report
    total_pairs = L_A * L_B
    num_contacts = int(contact_map.sum())
    num_interface_a = int(mask_a.sum())
    num_interface_b = int(mask_b.sum())
    pct_positive = (num_contacts / total_pairs) * 100.0 if total_pairs > 0 else 0.0

    print("\n==========================================================================")
    print(f" IRLM Dataset Builder Validation Report: {complex_id} ({chain_a} vs {chain_b})")
    print("==========================================================================")
    print(f" PDB ID                      : {complex_id}")
    print(f" Chain IDs                   : Partner A = '{chain_a}', Partner B = '{chain_b}'")
    print(f" Sequence Lengths            : L_A = {L_A}, L_B = {L_B}")
    print(f" ESM Embedding Shapes        : A = {emb_a.shape}, B = {emb_b.shape}")
    print(f" Contacting Residue Pairs    : {num_contacts} / {total_pairs}")
    print(f" Interface Residues in A     : {num_interface_a} / {L_A} ({num_interface_a/L_A*100:.1f}%)")
    print(f" Interface Residues in B     : {num_interface_b} / {L_B} ({num_interface_b/L_B*100:.1f}%)")
    print(f" Contact-Map Positives (%)   : {pct_positive:.2f}%")
    print(f" Saved Data File             : {out_file}")
    print("--------------------------------------------------------------------------")
    print(" Validation Checklist Status:")
    for check_name, status in validation_results.items():
        symbol = "PASS" if status else "FAIL"
        print(f"   [{symbol}] {check_name}")
    print("--------------------------------------------------------------------------")
    print(f" Overall Dataset Item Status  : {'SUCCESS - ALL CHECKS PASSED' if all_passed else 'FAILURE - VALIDATION ERROR'}")
    print("==========================================================================\n")

    return all_passed, out_file

def main():
    parser = argparse.ArgumentParser(description="Pilot IRLM Dataset Builder")
    parser.add_argument("--pdb", type=str, default="1YCR.pdb", help="Path to input PDB structure file")
    parser.add_argument("--chain-a", type=str, default="A", help="Chain ID for Partner A")
    parser.add_argument("--chain-b", type=str, default="B", help="Chain ID for Partner B")
    parser.add_argument("--complex-id", type=str, default="1YCR", help="Identifier for complex")
    parser.add_argument("--device", type=str, default="cpu", help="Device for ESM model (cpu or cuda)")
    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    if not pdb_path.is_absolute():
        pdb_path = PROJECT_ROOT / pdb_path

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Initializing ESMFeatureExtractor on device: {device}...")
    extractor = ESMFeatureExtractor(model_name="facebook/esm2_t12_35M_UR50D", device=device)

    success, saved_path = build_irlm_complex(
        pdb_path=pdb_path,
        chain_a=args.chain_a,
        chain_b=args.chain_b,
        complex_id=args.complex_id,
        extractor=extractor
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
