import torch
from typing import Dict, List, Any
import numpy as np

class HotspotAnalyzer:
    def __init__(self, sequence_model, esm_extractor, bio_manager=None, bio_encoder=None):
        self.seq_model = sequence_model
        self.esm_extractor = esm_extractor
        self.bio_manager = bio_manager
        self.bio_encoder = bio_encoder
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

        # 2. Vectorized Perturbation Function
        def compute_perturbations(target_seq, base_other_emb, is_p1=True):
            seq_len = len(target_seq)
            if seq_len < window_size:
                return [0.0] * seq_len

            # Adaptive step for responsive interactive performance on CPU
            step = max(1, (seq_len - window_size) // 30) if seq_len > 35 else 1

            windows = []
            masked_dict = {}
            for i in range(0, seq_len - window_size + 1, step):
                masked_seq = target_seq[:i] + "X" * window_size + target_seq[i+window_size:]
                wid = f"w_{i}"
                masked_dict[wid] = masked_seq
                windows.append((i, wid))

            if not windows:
                return [0.0] * seq_len

            # Extract all masked embeddings in parallel batches
            mut_embs = self.esm_extractor.get_embeddings(masked_dict, batch_size=16)

            mut_list = []
            for _, wid in windows:
                m_emb = mut_embs[wid].unsqueeze(0).to(self.device).float()
                if self.bio_manager and self.bio_encoder:
                    b_target = b1 if is_p1 else b2
                    m_emb = torch.cat([m_emb, b_target], dim=1)
                mut_list.append(m_emb)

            mut_batch = torch.cat(mut_list, dim=0)
            base_other_rep = base_other_emb.repeat(mut_batch.size(0), 1)

            with torch.no_grad():
                if is_p1:
                    raw_out = self.seq_model(mut_batch, base_other_rep)
                else:
                    raw_out = self.seq_model(base_other_rep, mut_batch)
                mut_scores = torch.sigmoid(raw_out).squeeze(-1).tolist()

            if isinstance(mut_scores, float):
                mut_scores = [mut_scores]

            res_impact = [0.0] * seq_len
            for (w_start, _), m_score in zip(windows, mut_scores):
                delta = max(0.0, base_score - m_score)
                w_end = min(seq_len, w_start + max(step, window_size))
                for j in range(w_start, w_end):
                    res_impact[j] = max(res_impact[j], delta)

            return res_impact

        total_p1 = compute_perturbations(p1_seq, e2_base, is_p1=True)
        total_p2 = compute_perturbations(p2_seq, e1_base, is_p1=False)

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
