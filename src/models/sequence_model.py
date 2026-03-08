import torch
import torch.nn as nn


class SequencePPIModel(nn.Module):
    def __init__(self, input_dim: int = 320, hidden_dim: int = 512, dropout: float = 0.3):
        """
        Enhanced MLP model for PPI prediction using protein embeddings.
        Uses concat + |diff| + hadamard for richer pair representations.
        Outputs raw logits (use BCEWithLogitsLoss during training).
        """
        super().__init__()
        # Input: [emb1, emb2, |emb1-emb2|, emb1*emb2] → input_dim * 4
        pair_dim = input_dim * 4

        self.input_proj = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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

        # Residual block 2
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection for residual skip (hidden_dim → hidden_dim // 2)
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1),
            # No Sigmoid — use BCEWithLogitsLoss for training
        )

    def forward(self, emb1, emb2):
        """
        Args:
            emb1: Tensor of shape (batch, input_dim)
            emb2: Tensor of shape (batch, input_dim)
        Returns:
            Raw logits of shape (batch, 1). Apply sigmoid yourself at inference.
        """
        # Rich pair features
        concat = torch.cat([emb1, emb2], dim=1)
        abs_diff = torch.abs(emb1 - emb2)
        hadamard = emb1 * emb2
        x = torch.cat([concat, abs_diff, hadamard], dim=1)

        # Forward with residual connections
        x = self.input_proj(x)              # → hidden_dim
        x = x + self.res1(x)                # residual block 1
        x = self.skip_proj(x) + self.res2(x)  # residual block 2 with dim reduction
        x = self.head(x)                    # → 1

        return x
