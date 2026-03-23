import torch
from typing import Dict, List, Any
import numpy as np

class HotspotAnalyzer:
    def __init__(self, sequence_model, esm_extractor):
        self.seq_model = sequence_model
        self.esm_extractor = esm_extractor
        self.device = next(sequence_model.parameters()).device

    def identify_hotspots(self, 
                          p1_id: str, p1_seq: str, 
                          p2_id: str, p2_seq: str, 
                          window_size: int = 5) -> Dict[str, Any]:
        """
        Identifies hotspots using a sliding window perturbation approach.
        """
        results = []
        
        # 1. Base Score
        with torch.no_grad():
            base_embs = self.esm_extractor.get_embeddings({p1_id: p1_seq, p2_id: p2_seq}, batch_size=2)
            e1_base = base_embs[p1_id].unsqueeze(0).to(self.device).float()
            e2_base = base_embs[p2_id].unsqueeze(0).to(self.device).float()
            base_score = torch.sigmoid(self.seq_model(e1_base, e2_base)).item()

        # 2. Perturb Protein 1
        scores_p1 = []
        for i in range(0, len(p1_seq) - window_size + 1):
            # Mask window with 'X'
            masked_seq = p1_seq[:i] + "X" * window_size + p1_seq[i+window_size:]
            
            with torch.no_grad():
                mut_embs = self.esm_extractor.get_embeddings({p1_id: masked_seq}, batch_size=1)
                e1_mut = mut_embs[p1_id].unsqueeze(0).to(self.device).float()
                mut_score = torch.sigmoid(self.seq_model(e1_mut, e2_base)).item()
                scores_p1.append(mut_score)
        
        # 3. Perturb Protein 2
        scores_p2 = []
        for i in range(0, len(p2_seq) - window_size + 1):
            masked_seq = p2_seq[:i] + "X" * window_size + p2_seq[i+window_size:]
            
            with torch.no_grad():
                mut_embs = self.esm_extractor.get_embeddings({p2_id: masked_seq}, batch_size=1)
                e2_mut = mut_embs[p2_id].unsqueeze(0).to(self.device).float()
                mut_score = torch.sigmoid(self.seq_model(e1_base, e2_mut)).item()
                scores_p2.append(mut_score)

        # 4. Process results
        def process_scores(scores, base, seq_len):
            deltas = [base - s for s in scores] # Higher delta = more critical (score dropped when masked)
            # Map window scores back to individual residues (max of windows containing residue)
            res_impact = [0.0] * seq_len
            for i, d in enumerate(deltas):
                for j in range(i, i + window_size):
                    if j < seq_len:
                        res_impact[j] = max(res_impact[j], d)
            return res_impact

        total_p1 = process_scores(scores_p1, base_score, len(p1_seq))
        total_p2 = process_scores(scores_p2, base_score, len(p2_seq))

        return {
            "base_score": base_score,
            "protein1": {
                "id": p1_id,
                "residue_impact": total_p1,
                "max_impact": max(total_p1) if total_p1 else 0
            },
            "protein2": {
                "id": p2_id,
                "residue_impact": total_p2,
                "max_impact": max(total_p2) if total_p2 else 0
            }
        }
