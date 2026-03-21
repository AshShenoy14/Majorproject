import torch
import torch.nn as nn


class SequencePPIModel(nn.Module):
    def __init__(self, input_dim: int = 480, bio_dim: int = 10, hidden_dim: int = 512, dropout: float = 0.3):
        """
        Enhanced MLP model with Biological Feature integration.
        input_dim: ESM embedding dimension.
        bio_dim: Biological feature dimension (e.g. localization).
        """
        super().__init__()
        # Total dimension per protein: ESM + Bio
        total_prot_dim = input_dim + bio_dim
        
        # Input: [emb1, emb2, |emb1-emb2|, emb1*emb2] → total_prot_dim * 4
        pair_dim = total_prot_dim * 4
        
        self.input_proj = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            # Use LayerNorm for better stability when combining disparate features
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual block 2 (Dimension Reduction)
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.skip_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, emb1, emb2):
        # Rich pair features
        concat = torch.cat([emb1, emb2], dim=1)
        abs_diff = torch.abs(emb1 - emb2)
        hadamard = emb1 * emb2
        x = torch.cat([concat, abs_diff, hadamard], dim=1)

        # Forward with residual connections
        x = self.input_proj(x)
        x = x + self.res1(x)
        x = self.skip_proj(x) + self.res2(x)
        x = self.head(x)

        return x
