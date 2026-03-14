import torch
import torch.nn as nn


class SequencePPIModel(nn.Module):
    """
    Sequence-based PPI predictor using ESM embeddings.

    Key design: Shared protein encoder projects each protein into a
    learned space BEFORE computing interaction features. This prevents
    collapse when raw ESM embeddings are too similar across proteins.

    Outputs raw logits — use BCEWithLogitsLoss for training,
    apply torch.sigmoid() only at inference.
    """

    def __init__(self, input_dim: int = 320, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        # Shared encoder — weight-tied across both proteins
        # Projects raw ESM embeddings into discriminative space
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Interaction classifier
        # Input: [h1 || h2 || h1*h2 || |h1-h2|] = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform + zero bias prevents logit drift at initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1: (batch, input_dim) mean-pooled ESM embedding for protein 1
            emb2: (batch, input_dim) mean-pooled ESM embedding for protein 2
        Returns:
            (batch, 1) raw logits
        """
        # Shared encoder processes each protein independently
        h1 = self.protein_encoder(emb1)
        h2 = self.protein_encoder(emb2)

        # Four interaction channels in learned space
        combined = torch.cat([
            h1,                       # protein 1 representation
            h2,                       # protein 2 representation
            h1 * h2,                  # element-wise product (co-activation)
            torch.abs(h1 - h2),       # absolute difference (complementarity)
        ], dim=1)

        return self.classifier(combined)