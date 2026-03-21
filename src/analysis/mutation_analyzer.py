import torch
import copy
from typing import Dict, Any, List

class MutationAnalyzer:
    def __init__(self, sequence_model, esm_extractor, bio_manager=None, bio_encoder=None):
        """
        Initialize with existing models and bio managers.
        """
        self.seq_model = sequence_model
        self.esm_extractor = esm_extractor
        self.bio_manager = bio_manager
        self.bio_encoder = bio_encoder
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
            
            # Add Bio Features
            if self.bio_manager and self.bio_encoder:
                meta = self.bio_manager.get_bio_metadata([p1_id, p2_id])
                def get_bio(pid):
                    row = meta[meta["protein_id"] == pid]
                    loc = row.iloc[0]["localization"] if not row.empty else ""
                    return self.bio_encoder.encode_protein(loc).to(self.device).float().unsqueeze(0)
                
                b1 = get_bio(p1_id)
                b2 = get_bio(p2_id)
                e1_base = torch.cat([e1_base, b1], dim=1)
                e2_base = torch.cat([e2_base, b2], dim=1)
            
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
            
            with torch.no_grad():
                if mut['protein'] == 1:
                    mut_embs = self.esm_extractor.get_embeddings({p1_id: mut_seq}, batch_size=1)
                    e1_mut = mut_embs[p1_id].unsqueeze(0).to(self.device).float()
                    if self.bio_manager and self.bio_encoder:
                        e1_mut = torch.cat([e1_mut, b1], dim=1)
                    mut_score = torch.sigmoid(self.seq_model(e1_mut, e2_base)).item()
                else:
                    mut_embs = self.esm_extractor.get_embeddings({p2_id: mut_seq}, batch_size=1)
                    e2_mut = mut_embs[p2_id].unsqueeze(0).to(self.device).float()
                    if self.bio_manager and self.bio_encoder:
                        e2_mut = torch.cat([e2_mut, b2], dim=1)
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
    def suggest_optimal_mutations(self, 
                                 p1_id: str, p1_seq: str, 
                                 p2_id: str, p2_seq: str, 
                                 mode: str = 'disrupt', # 'disrupt' or 'stabilize'
                                 top_n: int = 5) -> Dict[str, Any]:
        """
        Suggests mutations that most strongly disrupt or stabilize the interaction.
        Uses a heuristic search around predicted hotspots.
        """
        # 1. First, find hotspots to narrow search space
        from src.analysis.hotspot_analyzer import HotspotAnalyzer
        ha = HotspotAnalyzer(self.seq_model, self.esm_extractor, self.bio_manager, self.bio_encoder)
        hotspots = ha.identify_hotspots(p1_id, p1_seq, p2_id, p2_seq)
        
        # 2. Pick top positions based on impact
        impacts = hotspots['protein1']['residue_impact']
        # Get indices of top 3 impact positions
        top_positions = np.argsort(impacts)[-3:][::-1]
        
        candidates = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # 3. Test substitutions at these positions
        with torch.no_grad():
            base_embs = self.esm_extractor.get_embeddings({p1_id: p1_seq, p2_id: p2_seq}, batch_size=2)
            e1_base = base_embs[p1_id].unsqueeze(0).to(self.device).float()
            e2_base = base_embs[p2_id].unsqueeze(0).to(self.device).float()
            
            # Add Bio Features
            if self.bio_manager and self.bio_encoder:
                meta = self.bio_manager.get_bio_metadata([p1_id, p2_id])
                def get_bio(pid):
                    row = meta[meta["protein_id"] == pid]
                    loc = row.iloc[0]["localization"] if not row.empty else ""
                    return self.bio_encoder.encode_protein(loc).to(self.device).float().unsqueeze(0)
                
                b1 = get_bio(p1_id)
                b2 = get_bio(p2_id)
                e1_base = torch.cat([e1_base, b1], dim=1)
                e2_base = torch.cat([e2_base, b2], dim=1)
            
            base_score = torch.sigmoid(self.seq_model(e1_base, e2_base)).item()

            for pos in top_positions:
                orig_aa = p1_seq[pos]
                for mut_aa in amino_acids:
                    if mut_aa == orig_aa: continue
                    
                    # Create mutated sequence
                    mut_seq = p1_seq[:pos] + mut_aa + p1_seq[pos+1:]
                    mut_embs = self.esm_extractor.get_embeddings({p1_id: mut_seq}, batch_size=1)
                    e1_mut = mut_embs[p1_id].unsqueeze(0).to(self.device).float()
                    
                    if self.bio_manager and self.bio_encoder:
                        e1_mut = torch.cat([e1_mut, b1], dim=1)
                        
                    mut_score = torch.sigmoid(self.seq_model(e1_mut, e2_base)).item()
                    
                    candidates.append({
                        "pos": int(pos + 1),
                        "orig": orig_aa,
                        "mut": mut_aa,
                        "score": mut_score,
                        "delta": mut_score - base_score
                    })
        
        # 4. Sort and return
        if mode == 'disrupt':
            candidates.sort(key=lambda x: x['delta'])
        else:
            candidates.sort(key=lambda x: x['delta'], reverse=True)
            
        return {
            "mode": mode,
            "base_score": base_score,
            "suggestions": candidates[:top_n]
        }
