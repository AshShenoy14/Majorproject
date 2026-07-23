import torch
import numpy as np
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModel

class ResidueGraphGenerator:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cpu"):
        self.device = device
        print(f"Loading ESM-2 model: {model_name} on {device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        except Exception as e:
            print(f"Offline load failed ({e}), attempting regular load...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def generate_rig(self, sequence: str, threshold: float = 0.85, max_residues: int = 500, uniprot_id: str = None) -> Dict[str, Any]:
        """
        Generates a Residue Interaction Graph (RIG) for a given protein sequence.
        Nodes: Residues
        Edges: Sequence distance and 3D contact map (or embedding similarity if PDB not available)
        """
        # Truncate if too long for visualization performance
        if len(sequence) > max_residues:
            sequence = sequence[:max_residues]

        # 1. Get Residue-Level Embeddings
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # last_hidden_state shape: [1, seq_len + 2, hidden_dim] (includes BOS/EOS)
            residue_embeddings = outputs.last_hidden_state[0, 1:-1, :] # Strip BOS/EOS
        
        # 2. Calculate Cosine Similarity Matrix
        # Normalize for cosine similarity
        norm_embeddings = residue_embeddings / residue_embeddings.norm(dim=1, keepdim=True)
        sim_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # 3. Fetch AlphaFold 3D contacts if uniprot_id is provided
        use_3d = False
        pdb_contacts = []
        if uniprot_id:
            try:
                from src.data.alphafold_fetcher import AlphaFoldFetcher
                fetcher = AlphaFoldFetcher()
                pdb_path = fetcher.download_pdb(uniprot_id)
                if pdb_path:
                    coords = fetcher.parse_ca_coordinates(pdb_path)
                    if coords:
                        pdb_contacts = fetcher.calculate_contact_map(coords, threshold=8.0)
                        use_3d = True
            except Exception as e:
                print(f"Warning: Could not fetch AlphaFold contacts for {uniprot_id} ({e}), falling back to sequence similarity proxy.")

        # 4. Build Graph
        nodes = []
        for i, aa in enumerate(sequence):
            nodes.append({
                "id": str(i),
                "label": f"{aa}{i+1}",
                "residue": aa,
                "position": i + 1
            })

        edges = []
        seq_len = len(sequence)
        
        # Add primary chain edges
        for i in range(seq_len - 1):
            edges.append({
                "source": str(i),
                "target": str(i+1),
                "type": "backbone",
                "weight": 1.0
            })

        if use_3d and pdb_contacts:
            # Map 1-based residue numbers to 0-based indices and filter long-range contacts
            for r1, r2, dist in pdb_contacts:
                i, j = r1 - 1, r2 - 1
                if 0 <= i < seq_len and 0 <= j < seq_len:
                    if abs(i - j) >= 4:
                        edges.append({
                            "source": str(i),
                            "target": str(j),
                            "type": "3d_contact",
                            "weight": float(8.0 / (dist + 1e-5))
                        })
        else:
            # Add long-range interaction edges based on similarity
            sim_matrix_np = sim_matrix.cpu().numpy()
            for i in range(seq_len):
                for j in range(i + 4, seq_len): # Minimum 4 residues apart to be "long range"
                    score = float(sim_matrix_np[i, j])
                    if score > threshold:
                        edges.append({
                            "source": str(i),
                            "target": str(j),
                            "type": "contact_proxy",
                            "weight": score
                        })

        return {
            "nodes": nodes,
            "links": edges,
            "metadata": {
                "sequence_length": seq_len,
                "threshold": threshold,
                "num_contacts": len([e for e in edges if e['type'] in ('contact_proxy', '3d_contact')]),
                "use_3d": use_3d
            }
        }
