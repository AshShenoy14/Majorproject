import torch
import copy
from typing import Dict, Any, List

class MutationAnalyzer:
    def __init__(self, sequence_model, esm_extractor):
        """
        Initialize with existing models.
        """
        self.seq_model = sequence_model
        self.esm_extractor = esm_extractor
        self.device = next(sequence_model.parameters()).device

    def project_mutation_impact(self, 
                               p1_id: str, p1_seq: str, 
                               p2_id: str, p2_seq: str, 
                               mutations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predicts the impact of mutations on the interaction score.
        
        mutations: List of dicts like {'protein': 1, 'pos': 124, 'orig': 'S', 'mut': 'A'}
        """
        results = []
        
        # 1. Base Score
        with torch.no_grad():
            base_embs = self.esm_extractor.get_embeddings({p1_id: p1_seq, p2_id: p2_seq}, batch_size=2)
            e1_base = base_embs[p1_id].unsqueeze(0).to(self.device).float()
            e2_base = base_embs[p2_id].unsqueeze(0).to(self.device).float()
            base_score = torch.sigmoid(self.seq_model(e1_base, e2_base)).item()

        # 2. Mutated Scores
        for mut in mutations:
            target_p = p1_id if mut['protein'] == 1 else p2_id
            target_seq = p1_seq if mut['protein'] == 1 else p2_seq
            pos = mut['pos'] - 1 # 1-indexed to 0-indexed
            
            if pos < 0 or pos >= len(target_seq):
                results.append({**mut, "error": "Position out of bounds", "impact": 0})
                continue
                
            if target_seq[pos] != mut['orig']:
                results.append({**mut, "error": f"Original residue mismatch (found {target_seq[pos]} at {mut['pos']})", "impact": 0})
                continue
            
            # Create mutated sequence
            mut_seq = target_seq[:pos] + mut['mut'] + target_seq[pos+1:]
            
            # Recalculate embeddings and score
            with torch.no_grad():
                if mut['protein'] == 1:
                    mut_embs = self.esm_extractor.get_embeddings({p1_id: mut_seq}, batch_size=1)
                    e1_mut = mut_embs[p1_id].unsqueeze(0).to(self.device).float()
                    mut_score = torch.sigmoid(self.seq_model(e1_mut, e2_base)).item()
                else:
                    mut_embs = self.esm_extractor.get_embeddings({p2_id: mut_seq}, batch_size=1)
                    e2_mut = mut_embs[p2_id].unsqueeze(0).to(self.device).float()
                    mut_score = torch.sigmoid(self.seq_model(e1_base, e2_mut)).item()
            
            impact = mut_score - base_score
            results.append({
                **mut,
                "base_score": base_score,
                "mutated_score": mut_score,
                "impact_delta": impact,
                "interpretation": "Disruptive" if impact < -0.05 else ("Enhancing" if impact > 0.05 else "Neutral")
            })
            
        return {
            "protein1": p1_id,
            "protein2": p2_id,
            "mutation_results": results
        }
