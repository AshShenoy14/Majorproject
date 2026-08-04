import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, List, Any, Optional

class BiDirectionalCrossAttention(nn.Module):
    """
    Bi-Directional Multi-Head Cross-Attention over unpooled residue embeddings.
    Computes a symmetric residue-residue interaction matrix M_ij between Protein A (L_A) and Protein B (L_B).
    """
    def __init__(self, embed_dim: int = 480, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_a = nn.Linear(embed_dim, embed_dim)
        self.k_proj_b = nn.Linear(embed_dim, embed_dim)
        self.v_proj_b = nn.Linear(embed_dim, embed_dim)

        self.q_proj_b = nn.Linear(embed_dim, embed_dim)
        self.k_proj_a = nn.Linear(embed_dim, embed_dim)
        self.v_proj_a = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_a: [L_A, embed_dim]
            h_b: [L_B, embed_dim]
        Returns:
            attn_ab: [L_A, L_B] attention from A -> B
            attn_ba: [L_B, L_A] attention from B -> A
            interaction_matrix: [L_A, L_B] symmetric interaction score map M_ij
        """
        L_A, D = h_a.shape
        L_B, _ = h_b.shape

        # Unsqueeze batch dimension for torch operations: [1, L, D]
        h_a_b = h_a.unsqueeze(0)
        h_b_b = h_b.unsqueeze(0)

        # 1. A queries B
        q_a = self.q_proj_a(h_a_b).view(1, L_A, self.num_heads, self.head_dim).transpose(1, 2) # [1, H, L_A, d_k]
        k_b = self.k_proj_b(h_b_b).view(1, L_B, self.num_heads, self.head_dim).transpose(1, 2) # [1, H, L_B, d_k]

        scores_ab = torch.matmul(q_a, k_b.transpose(-2, -1)) * self.scale # [1, H, L_A, L_B]
        attn_ab_heads = F.softmax(scores_ab, dim=-1) # [1, H, L_A, L_B]

        # 2. B queries A
        q_b = self.q_proj_b(h_b_b).view(1, L_B, self.num_heads, self.head_dim).transpose(1, 2) # [1, H, L_B, d_k]
        k_a = self.k_proj_a(h_a_b).view(1, L_A, self.num_heads, self.head_dim).transpose(1, 2) # [1, H, L_A, d_k]

        scores_ba = torch.matmul(q_b, k_a.transpose(-2, -1)) * self.scale # [1, H, L_B, L_A]
        attn_ba_heads = F.softmax(scores_ba, dim=-1) # [1, H, L_B, L_A]

        # 3. Form head-averaged bidirectional interaction matrix
        # attn_ab: [L_A, L_B], attn_ba: [L_B, L_A] -> transpose to [L_A, L_B]
        attn_ab = attn_ab_heads.mean(dim=1).squeeze(0) # [L_A, L_B]
        attn_ba = attn_ba_heads.mean(dim=1).squeeze(0).transpose(0, 1) # [L_A, L_B]

        # Geometric mean of forward and backward cross-attentions
        interaction_matrix = torch.sqrt(torch.clamp(attn_ab * attn_ba, min=1e-8))

        return attn_ab, attn_ba.transpose(0, 1), interaction_matrix


class GraphResidueGating(nn.Module):
    """
    Modulates residue cross-attention scores using graph node features or embeddings.
    Allows graph topology to guide sequence-level interaction localization.
    """
    def __init__(self, seq_dim: int = 480, graph_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor, z_a: Optional[torch.Tensor] = None, z_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h_a: [L_A, seq_dim]
            h_b: [L_B, seq_dim]
            z_a: [graph_dim] or None
            z_b: [graph_dim] or None
        Returns:
            gate_matrix: [L_A, L_B] tensor in range (0, 1)
        """
        L_A, _ = h_a.shape
        L_B, _ = h_b.shape

        h_a_proj = self.seq_proj(h_a) # [L_A, hidden_dim]
        h_b_proj = self.seq_proj(h_b) # [L_B, hidden_dim]

        if z_a is not None and z_b is not None:
            z_a_proj = self.graph_proj(z_a.squeeze(0)) # [hidden_dim]
            z_b_proj = self.graph_proj(z_b.squeeze(0)) # [hidden_dim]
            # Combine graph context with sequence features
            h_a_proj = h_a_proj + z_a_proj
            h_b_proj = h_b_proj + z_b_proj

        # Outer Hadamard/Concatenation representation for pairs
        # Broadcast [L_A, 1, hidden] and [1, L_B, hidden]
        a_exp = h_a_proj.unsqueeze(1).expand(L_A, L_B, -1)
        b_exp = h_b_proj.unsqueeze(0).expand(L_A, L_B, -1)

        pair_feat = torch.cat([a_exp * b_exp, torch.abs(a_exp - b_exp)], dim=-1) # [L_A, L_B, 2*hidden_dim]
        gate_matrix = self.gate_mlp(pair_feat).squeeze(-1) # [L_A, L_B]

        return gate_matrix


class Gaussian1DSmoother(nn.Module):
    """
    1D Gaussian kernel smoothing over 1D sequence residue scores.
    Removes high-frequency noise while preserving contiguous physical interaction peaks.
    """
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - self.padding
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        # Shape for Conv1d: [out_channels, in_channels/groups, kernel_size] -> [1, 1, K]
        self.register_buffer("kernel", kernel.view(1, 1, kernel_size))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [L] 1D tensor of residue importance scores
        Returns:
            smoothed_scores: [L] 1D tensor
        """
        if len(scores) < self.kernel_size:
            return scores
        x = scores.view(1, 1, -1) # [1, 1, L]
        smoothed = F.conv1d(x, self.kernel, padding=self.padding)
        return smoothed.squeeze()


class InteractionRegionLocalizationModule(nn.Module):
    """
    Interaction Region Localization Module (IRLM).
    Algorithmic PyTorch module that combines:
    1. Unpooled ESM-2 residue embeddings
    2. Bi-directional Multi-Head Cross-Attention
    3. Graph-gated feature modulation
    4. 1D Gaussian kernel smoothing
    5. Contiguous region detection and peak extraction algorithm
    """
    def __init__(self, 
                 embed_dim: int = 480, 
                 graph_dim: int = 256,
                 num_heads: int = 8, 
                 alpha: float = 0.6,
                 gap_max: int = 3,
                 smooth_sigma: float = 1.2,
                 device: Optional[str] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.gap_max = gap_max

        self.cross_attn = BiDirectionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.graph_gating = GraphResidueGating(seq_dim=embed_dim, graph_dim=graph_dim)
        self.smoother = Gaussian1DSmoother(kernel_size=5, sigma=smooth_sigma)

        # Region importance scoring MLP
        self.importance_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def compute_residue_importance(self, 
                                   h_a: torch.Tensor, 
                                   h_b: torch.Tensor, 
                                   z_a: Optional[torch.Tensor] = None, 
                                   z_b: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes 2D residue interaction matrix M_ij and 1D per-residue importance scores R_A, R_B.
        """
        L_A = h_a.shape[0]
        L_B = h_b.shape[0]

        # 1. Cross-attention matrices
        _, _, attn_matrix = self.cross_attn(h_a, h_b) # [L_A, L_B]

        # 2. Graph gating modulation
        gate_matrix = self.graph_gating(h_a, h_b, z_a, z_b) # [L_A, L_B]
        interaction_matrix = attn_matrix * gate_matrix # [L_A, L_B]

        # 3. Self-importance intrinsic modulation via importance head
        self_imp_a = self.importance_head(h_a).squeeze(-1) # [L_A]
        self_imp_b = self.importance_head(h_b).squeeze(-1) # [L_B]

        # 4. Aggregation across sequence dimensions: max-pool + mean-pool
        max_a, _ = torch.max(interaction_matrix, dim=1) # [L_A]
        mean_a = torch.mean(interaction_matrix, dim=1) # [L_A]
        r_a_raw = (self.alpha * max_a + (1.0 - self.alpha) * mean_a) * self_imp_a

        max_b, _ = torch.max(interaction_matrix, dim=0) # [L_B]
        mean_b = torch.mean(interaction_matrix, dim=0) # [L_B]
        r_b_raw = (self.alpha * max_b + (1.0 - self.alpha) * mean_b) * self_imp_b

        # 5. Gaussian Kernel Smoothing
        r_a_smooth = self.smoother(r_a_raw)
        r_b_smooth = self.smoother(r_b_raw)

        # 6. Min-Max Normalization to [0, 1]
        def min_max_norm(x):
            min_val = torch.min(x)
            max_val = torch.max(x)
            if max_val - min_val < 1e-6:
                return torch.zeros_like(x)
            return (x - min_val) / (max_val - min_val + 1e-8)

        r_a_norm = min_max_norm(r_a_smooth)
        r_b_norm = min_max_norm(r_b_smooth)

        return r_a_norm, r_b_norm, interaction_matrix

    def extract_interaction_regions(self, 
                                     importance_scores: torch.Tensor, 
                                     top_percentile: float = 0.80) -> Tuple[List[int], List[int], float]:
        """
        Algorithmic region merging and peak extraction.
        Args:
            importance_scores: [L] 1D tensor normalized in [0, 1]
            top_percentile: percentile threshold for region detection
        Returns:
            region: [start_1indexed, end_1indexed]
            key_residues: list of 1-indexed local peak positions
            region_ratio: fraction of importance energy contained within region
        """
        scores_np = importance_scores.detach().cpu().numpy()
        L = len(scores_np)
        if L == 0:
            return [1, 1], [1], 1.0

        # Dynamic Threshold calculation
        threshold = float(torch.quantile(importance_scores, top_percentile).item())
        threshold = max(threshold, 0.40) # Ensure a reasonable baseline floor

        # Find binary active positions
        active = (scores_np >= threshold)

        if not active.any():
            # Fallback to top-3 peak indices if no residue exceeds threshold
            top_idx = int(scores_np.argmax())
            start = max(1, top_idx - 2 + 1)
            end = min(L, top_idx + 2 + 1)
            return [start, end], [top_idx + 1], 1.0

        # Extract contiguous segments
        segments = []
        in_segment = False
        curr_start = 0

        for i in range(L):
            if active[i] and not in_segment:
                in_segment = True
                curr_start = i
            elif not active[i] and in_segment:
                in_segment = False
                segments.append((curr_start, i - 1))
        if in_segment:
            segments.append((curr_start, L - 1))

        # Merge segments separated by gap <= self.gap_max
        merged_segments = []
        if segments:
            curr_s, curr_e = segments[0]
            for s, e in segments[1:]:
                if s - curr_e <= self.gap_max:
                    curr_e = e # Merge
                else:
                    merged_segments.append((curr_s, curr_e))
                    curr_s, curr_e = s, e
            merged_segments.append((curr_s, curr_e))

        # Pick segment with the highest aggregated energy
        best_seg = max(merged_segments, key=lambda seg: scores_np[seg[0]:seg[1]+1].sum())
        s_idx, e_idx = best_seg

        # Convert to 1-indexed range
        region_1indexed = [s_idx + 1, e_idx + 1]

        # Extract local peaks inside best segment for key residues
        key_residues_1indexed = []
        seg_scores = scores_np[s_idx:e_idx+1]
        for idx_local in range(len(seg_scores)):
            global_idx = s_idx + idx_local
            is_left_higher = (idx_local == 0 or seg_scores[idx_local] >= seg_scores[idx_local - 1])
            is_right_higher = (idx_local == len(seg_scores) - 1 or seg_scores[idx_local] >= seg_scores[idx_local + 1])
            if is_left_higher and is_right_higher and scores_np[global_idx] >= threshold:
                key_residues_1indexed.append(global_idx + 1)

        if not key_residues_1indexed:
            # Fallback: top residue in segment
            best_local = int(seg_scores.argmax())
            key_residues_1indexed.append(s_idx + best_local + 1)

        # Region confidence ratio
        total_energy = scores_np.sum() + 1e-8
        region_energy = scores_np[s_idx:e_idx+1].sum()
        region_ratio = float(region_energy / total_energy)

        return region_1indexed, key_residues_1indexed[:5], region_ratio

    def forward(self, 
                h_a: torch.Tensor, 
                h_b: torch.Tensor, 
                z_a: Optional[torch.Tensor] = None, 
                z_b: Optional[torch.Tensor] = None,
                base_interaction_prob: float = 0.5) -> Dict[str, Any]:
        """
        Forward execution of IRLM during inference.
        Returns complete structured dictionary.
        """
        r_a, r_b, interaction_matrix = self.compute_residue_importance(h_a, h_b, z_a, z_b)

        reg_a, keys_a, ratio_a = self.extract_interaction_regions(r_a)
        reg_b, keys_b, ratio_b = self.extract_interaction_regions(r_b)

        # Overall Region Confidence score [0, 1]
        # Combines energy ratio with base interaction probability confidence
        region_confidence = min(0.99, max(0.50, round((ratio_a + ratio_b) / 2.0 * (0.5 + 0.5 * base_interaction_prob), 2)))

        return {
            "interaction_probability": float(base_interaction_prob),
            "protein_A_region": reg_a,
            "protein_B_region": reg_b,
            "protein_A_key_residues": keys_a,
            "protein_B_key_residues": keys_b,
            "region_confidence": float(region_confidence),
            "protein_A_importance_scores": [round(float(v), 4) for v in r_a.detach().cpu().tolist()],
            "protein_B_importance_scores": [round(float(v), 4) for v in r_b.detach().cpu().tolist()],
        }


class IRLMLoss(nn.Module):
    """
    Composite loss function for training/finetuning IRLM.
    Combines:
    1. Binary Cross Entropy Loss on overall interaction probability
    2. L1 Sparsity Loss to enforce localized, non-diffuse binding interface
    3. Smoothness Loss to encourage spatially contiguous residue regions
    """
    def __init__(self, lambda_sparsity: float = 0.05, lambda_smooth: float = 0.05):
        super().__init__()
        self.bce = nn.BCELoss()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smooth = lambda_smooth

    def forward(self, 
                pred_prob: torch.Tensor, 
                target_label: torch.Tensor, 
                interaction_matrix: torch.Tensor, 
                r_a: torch.Tensor, 
                r_b: torch.Tensor) -> torch.Tensor:

        # 1. Main task loss
        loss_bce = self.bce(pred_prob, target_label)

        # 2. Sparsity Loss on 2D matrix M_ij
        loss_sparsity = torch.mean(torch.abs(interaction_matrix))

        # 3. Contiguous Smoothness Loss on 1D importance profiles
        diff_a = r_a[1:] - r_a[:-1] if len(r_a) > 1 else torch.tensor(0.0, device=r_a.device)
        diff_b = r_b[1:] - r_b[:-1] if len(r_b) > 1 else torch.tensor(0.0, device=r_b.device)
        loss_smooth = torch.mean(diff_a ** 2) + torch.mean(diff_b ** 2)

        return loss_bce + self.lambda_sparsity * loss_sparsity + self.lambda_smooth * loss_smooth
