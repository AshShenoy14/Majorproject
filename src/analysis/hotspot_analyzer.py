import torch
from typing import Callable, Dict, List, Any, Optional, Sequence

# progress(done, total, stage) is called as ESM passes complete; used by the backend's background jobs.
ProgressFn = Optional[Callable[[int, int, str], None]]

CHUNK = 8  # masked sequences per ESM batch (and per progress update)


def window_starts(seq_len: int, window_size: int = 5) -> List[int]:
    """Start positions of the masked windows: every residue for short sequences, ~30 evenly spaced windows otherwise."""
    if seq_len < window_size:
        return []
    step = max(1, (seq_len - window_size) // 30) if seq_len > 35 else 1
    return list(range(0, seq_len - window_size + 1, step))


class HotspotAnalyzer:
    def __init__(self, sequence_model, esm_extractor, bio_manager=None, bio_encoder=None):
        self.seq_model = sequence_model
        self.esm_extractor = esm_extractor
        self.bio_manager = bio_manager
        self.bio_encoder = bio_encoder
        self.device = next(sequence_model.parameters()).device

    def count_passes(self, p1_seq: str, p2_seq: str, proteins: Sequence[int] = (1, 2), window_size: int = 5) -> int:
        """Number of masked ESM passes identify_hotspots() will run (for progress reporting)."""
        seqs = {1: p1_seq, 2: p2_seq}
        return sum(len(window_starts(len(seqs[p]), window_size)) for p in proteins)

    def identify_hotspots(self,
                          p1_id: str, p1_seq: str,
                          p2_id: str, p2_seq: str,
                          window_size: int = 5,
                          proteins: Sequence[int] = (1, 2),
                          progress: ProgressFn = None) -> Dict[str, Any]:
        """
        Identifies hotspots by occlusion: each window of `window_size` residues is masked ('X'), the protein is
        re-embedded with ESM-2, and the drop in the sequence model's interaction score is attributed to the
        residues in that window. `proteins` limits the scan to protein 1 and/or 2.
        """
        # 1. Base Score
        with torch.no_grad():
            base_embs = self.esm_extractor.get_embeddings({p1_id: p1_seq, p2_id: p2_seq}, batch_size=2)
            e1_base = base_embs[p1_id].unsqueeze(0).to(self.device).float()
            e2_base = base_embs[p2_id].unsqueeze(0).to(self.device).float()

            # Add Bio Features
            b1 = b2 = None
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

        total = self.count_passes(p1_seq, p2_seq, proteins, window_size)
        done = 0

        def compute_perturbations(target_seq, base_other_emb, is_p1=True):
            nonlocal done
            seq_len = len(target_seq)
            starts = window_starts(seq_len, window_size)
            if not starts:
                return [0.0] * seq_len
            step = starts[1] - starts[0] if len(starts) > 1 else 1

            mut_scores = []
            for c in range(0, len(starts), CHUNK):
                chunk = starts[c:c + CHUNK]
                masked = {f"w_{i}": target_seq[:i] + "X" * window_size + target_seq[i + window_size:] for i in chunk}
                embs = self.esm_extractor.get_embeddings(masked, batch_size=CHUNK)
                batch = torch.stack([embs[f"w_{i}"].to(self.device).float() for i in chunk])
                if b1 is not None:
                    batch = torch.cat([batch, (b1 if is_p1 else b2).repeat(len(chunk), 1)], dim=1)
                other = base_other_emb.repeat(len(chunk), 1)
                with torch.no_grad():
                    raw_out = self.seq_model(batch, other) if is_p1 else self.seq_model(other, batch)
                mut_scores.extend(torch.sigmoid(raw_out).reshape(-1).tolist())
                done += len(chunk)
                if progress:
                    progress(done, total, f"Masking windows of protein {1 if is_p1 else 2}")

            res_impact = [0.0] * seq_len
            for w_start, m_score in zip(starts, mut_scores):
                delta = max(0.0, base_score - m_score)
                w_end = min(seq_len, w_start + max(step, window_size))
                for j in range(w_start, w_end):
                    res_impact[j] = max(res_impact[j], delta)
            return res_impact

        result = {"base_score": base_score}
        for p, pid, seq, other, is_p1 in ((1, p1_id, p1_seq, e2_base, True), (2, p2_id, p2_seq, e1_base, False)):
            if p in proteins:
                impact = compute_perturbations(seq, other, is_p1)
                result[f"protein{p}"] = {"id": pid, "residue_impact": impact, "max_impact": max(impact) if impact else 0}
        return result
