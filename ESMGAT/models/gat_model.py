import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm as GNNBatchNorm


class GATLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 643,
        hidden_channels: int = 256,
        heads: int = 4,
        dropout: float = 0.4,
    ):
        """
        Graph Attention Network (GAT) encoder + bilinear/MLP decoder for link prediction.
        
        Encoder:
          Two multi-head GATConv layers with self-attention aggregation.
          Layer 1: in_channels (483) -> (heads * head_dim) = (4 * 64) = 256 (concat=True)
          Layer 2: hidden_channels (256) -> hidden_channels (256) with 4 heads (concat=False)
          Activations and batch norms match the existing GraphSAGE implementation exactly (GNNBatchNorm, ReLU, Dropout).

        Decoder Head:
          Identical to SAGELinkPredictor for fair scientific comparison:
          Features: [u, v, |u - v|, u * v, bilinear(u, v)] -> (hidden_channels * 4) + 1 = 1025
          Classifier: Linear(1025, 512) -> BatchNorm1d -> GELU -> Dropout(0.4) -> Linear(512, 256) -> GELU -> Linear(256, 1)
        """
        super().__init__()
        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(in_channels)

        # GAT Encoding Layers
        head_dim = hidden_channels // heads  # 256 // 4 = 64
        self.conv1 = GATConv(
            in_channels, head_dim, heads=heads, concat=True, dropout=dropout
        )
        self.bn1 = GNNBatchNorm(hidden_channels)

        self.conv2 = GATConv(
            hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout
        )
        self.bn2 = GNNBatchNorm(hidden_channels)

        # Decoder Head: EXACT parity with GraphSAGE (SAGELinkPredictor)
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        classifier_input_dim = (hidden_channels * 4) + 1  # (256 * 4) + 1 = 1025

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def encode(self, x, edge_index):
        x = self.input_norm(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def encode_with_attention(self, x, edge_index):
        """
        Same computation as encode(), but also returns each GATConv layer's attention coefficients.

        Returns (z, [(edge_index_1, alpha_1), (edge_index_2, alpha_2)]), where edge_index_l includes the self-loops
        GATConv adds and alpha_l has shape [num_edges, heads]. alpha_l[e, h] is the weight target node
        edge_index_l[1, e] gives source node edge_index_l[0, e] in head h; weights over a node's incoming edges
        sum to 1 per head.
        """
        x = self.input_norm(x)
        x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, [att1, att2]

    def decode(self, z, src, dst):
        h_u, h_v = z[src], z[dst]
        bilinear_out = self.bilinear(h_u, h_v)
        diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        pair_repr = torch.cat([h_u, h_v, diff, hadamard, bilinear_out], dim=1)
        return self.classifier(pair_repr)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        src, dst = edge_label_index
        return self.decode(z, src, dst)
