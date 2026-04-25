import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINConv, BatchNorm as GNNBatchNorm

class GNNLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, dropout: float = 0.4):
        """
        High-Performance GraphSAGE for inductive link prediction.
        Focuses on neighborhood aggregation for 85%+ accuracy.
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        
        # Encoding layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = GNNBatchNorm(hidden_channels)
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = GNNBatchNorm(hidden_channels)
        
        self.dropout = dropout

        # Decoder Head
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        # Features: [u, v, |u-v|, u*v, bilinear]
        classifier_input_dim = (hidden_channels * 4) + 1
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
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


class GINLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 3, dropout: float = 0.4):
        """
        Graph Isomorphism Network (GIN) for link prediction.
        Matches the architecture of previously trained checkpoints.
        """
        super().__init__()
        
        # Input normalization (LayerNorm on raw features)
        self.input_norm = nn.LayerNorm(in_channels)
        
        # Input projection to hidden_channels (Linear + LayerNorm)
        self.pre_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        
        # GIN Convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(GNNBatchNorm(hidden_channels))
        
        self.dropout = dropout

        # Decoder Head
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        # Features: [u, v, |u-v|, u*v, bilinear]
        classifier_input_dim = (hidden_channels * 4) + 1
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def encode(self, x, edge_index):
        x = self.input_norm(x)
        x = self.pre_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

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


# Backward compatibility alias
GATLinkPredictor = GNNLinkPredictor
