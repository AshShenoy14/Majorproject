import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATLinkPredictor(nn.Module):
    """
    GAT-based link predictor for PPI graphs.
    
    Outputs raw logits — use BCEWithLogitsLoss for training,
    apply torch.sigmoid() only at inference.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads,
            dropout=dropout, add_self_loops=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(
            hidden_channels * heads, hidden_channels, heads=1,
            concat=False, dropout=dropout, add_self_loops=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Input: [src || dst || src*dst || |src-dst|] = hidden_channels * 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, 1),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode all nodes using 2-layer GAT."""
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Score edges using precomputed node embeddings."""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        combined = torch.cat([
            src, dst, src * dst, torch.abs(src - dst)
        ], dim=1)
        return self.classifier(combined)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        """Full forward: encode nodes then score edges."""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)