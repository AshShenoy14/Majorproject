import pytest
from pathlib import Path
from src.data.alphafold_fetcher import AlphaFoldFetcher

def test_alphafold_fetcher_integration():
    # Setup fetcher with default cache directory
    fetcher = AlphaFoldFetcher()
    
    # We'll use a small human protein: P62979 (Ubiquitin-40S ribosomal protein S27a)
    uniprot_id = "P62979"
    
    # Download PDB
    pdb_path = fetcher.download_pdb(uniprot_id)
    assert pdb_path is not None
    assert pdb_path.exists()
    
    # Parse CA Coordinates
    coords = fetcher.parse_ca_coordinates(pdb_path)
    assert len(coords) > 0
    
    # Check shape/keys
    first_coord = coords[0]
    assert "res_seq" in first_coord
    assert "x" in first_coord
    assert "y" in first_coord
    assert "z" in first_coord
    
    # Parse CB Coordinates
    cb_coords = fetcher.parse_cb_coordinates(pdb_path)
    assert len(cb_coords) > 0
    assert len(cb_coords) == len(coords)  # since all residues should map to either CB or CA fallback
    first_cb = cb_coords[0]
    assert "res_seq" in first_cb
    assert "x" in first_cb
    assert "y" in first_cb
    assert "z" in first_cb
    
    # Calculate Contact Map (standard 8.0 A threshold)
    contacts = fetcher.calculate_contact_map(coords, threshold=8.0)
    assert len(contacts) > 0
    
    # Check contact properties
    first_contact = contacts[0]
    assert len(first_contact) == 3  # (res1, res2, distance)
    assert first_contact[2] <= 8.0
