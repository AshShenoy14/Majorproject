import torch
import torch.nn as nn
import torch.nn.functional as F

class SequencePPIModel(nn.Module):
    def __init__(self, input_dim=480, hidden_dim=1024, dropout=0.3):
        """
        Ultra-High Capacity Symmetric MLP for Sequence-Based PPI Prediction.
        Designed to reach 92%+ base accuracy.
        """
        super().__init__()
        self.feature_dim = input_dim * 4
        
        # Input layer
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deep Residual Architecture
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        self.skip_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, emb1, emb2):
        # ESM-2 embeddings (480 dims)
        emb1 = emb1[:, :480]
        emb2 = emb2[:, :480]

        # Symmetric operators for orientation-invariance
        f_sum = emb1 + emb2
        f_prod = emb1 * emb2
        f_diff = torch.abs(emb1 - emb2)
        f_max = torch.max(emb1, emb2)
        
        x = torch.cat([f_sum, f_prod, f_diff, f_max], dim=1)
        
        # Forward pass with residuals
        x = self.input_proj(x)
        identity = x
        x = F.gelu(self.res1(x) + identity)
        
        identity = self.skip_proj(x)
        x = F.gelu(self.res2(x) + identity)
        
        return self.head(x)
