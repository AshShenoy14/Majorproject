import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATLinkPredictor(nn.Module):
    """
    GAT-based link predictor that works both transductively
    (proteins in graph) and inductively (new proteins).
    
    Key change: sequence_fallback encoder handles unseen proteins.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # GAT encoder (for proteins WITH graph context)
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

        # Fallback encoder (for proteins WITHOUT graph context)
        # Maps raw ESM embeddings → same space as GAT output
        self.sequence_fallback = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
        )

        # Edge classifier: [src || dst || src*dst || |src-dst|]
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

        self._init_weights()

    def _init_weights(self):
        for module in [self.classifier, self.sequence_fallback]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes using 2-layer GAT."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        return h

    def encode_sequences(self, x: torch.Tensor) -> torch.Tensor:
        """Encode proteins using only their ESM embeddings (no graph)."""
        return self.sequence_fallback(x)

    def decode(self, z: torch.Tensor,
               edge_label_index: torch.Tensor) -> torch.Tensor:
        """Score edges using node embeddings."""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        combined = torch.cat([
            src, dst, src * dst, torch.abs(src - dst)
        ], dim=1)
        return self.classifier(combined)

    def decode_from_embeddings(self, z_src: torch.Tensor,
                                z_dst: torch.Tensor) -> torch.Tensor:
        """Score edges from pre-computed embedding pairs (for inference)."""
        combined = torch.cat([
            z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst)
        ], dim=1)
        return self.classifier(combined)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        """Standard forward: encode all nodes, then score edges."""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)