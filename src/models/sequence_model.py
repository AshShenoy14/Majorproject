import torch
import torch.nn as nn

class SequencePPIModel(nn.Module):
    def __init__(self, input_dim: int = 320, hidden_dim: int = 256, dropout: float = 0.2):
        """
        MLP model for PPI prediction using concatenated protein embeddings.
        Input dim default 320 for esm2_t6_8M_UR50D.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        """
        Args:
            emb1: Tensor of shape (batch, input_dim)
            emb2: Tensor of shape (batch, input_dim)
        """
        x = torch.cat([emb1, emb2], dim=1)
        return self.net(x)
