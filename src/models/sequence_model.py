import torch
import torch.nn as nn


class SequencePPIModel(nn.Module):
    def __init__(self, input_dim: int = 480, hidden_dim: int = 768, dropout: float = 0.3):
        """
        Stable Symmetric MLP for PPI Prediction.
        Focuses on ESM embeddings as biological features are currently sparse.
        """
        super().__init__()
        # Symmetric Features: [A+B, A*B, |A-B|, max(A,B)]
        # This covers almost all pairwise relationships
        self.feature_dim = input_dim * 4
        
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual Blocks
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.skip_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, emb1, emb2):
        # Ensure we only use the ESM part if bio was appended
        # ESM-2 8M is 480 dims
        emb1 = emb1[:, :480]
        emb2 = emb2[:, :480]

        # Symmetric operators
        f_sum = emb1 + emb2
        f_prod = emb1 * emb2
        f_diff = torch.abs(emb1 - emb2)
        f_max = torch.max(emb1, emb2)
        
        x = torch.cat([f_sum, f_prod, f_diff, f_max], dim=1)
        
        x = self.input_proj(x)
        x = x + self.res1(x)
        x = self.skip_proj(x) + self.res2(x)
        
        return self.head(x)
