import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm as GNNBatchNorm

class GNNLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, heads: int = 4, dropout: float = 0.5):
        """
        Research-Grade GATv2 with Structural Bottlenecking and Bilinear Interaction (GATv3).
        Designed to reach 85%+ ROC-AUC by focusing on shared topology over raw sequence.
        """
        super().__init__()
        
        # 0. Input Normalization: Handle high-variance ESM features
        self.input_norm = nn.LayerNorm(in_channels)
        
        # 1. Structural Bottleneck: Compress high-dim ESM (480+) into structural vector
        self.pre_proj = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Message Passing Layers (GATv2)
        # Using 128 as compressed input dim
        self.conv1 = GATv2Conv(128, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = GNNBatchNorm(hidden_channels * heads)
        
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = GNNBatchNorm(hidden_channels * heads)
        
        # Skip connection from compressed input
        self.skip1 = nn.Linear(128, hidden_channels * heads)
        self.dropout = dropout

        # 3. Bilinear Interaction Head (Bio-Inspired Link Prediction)
        # Instead of just concatenating, we measure the direct relationship between u and v
        gnn_out_dim = hidden_channels * heads
        self.bilinear = nn.Bilinear(gnn_out_dim, gnn_out_dim, 1)
        
        # 4. Final Fusion Head: [u, v, |u-v|, u*v, bilinear(u,v)]
        classifier_input_dim = (gnn_out_dim * 4) + 1
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(256, 1)
        )
        # Initialize final bias to negative to prevent "all-ones" collapse at start
        nn.init.constant_(self.classifier[-1].bias, -2.0)

    def encode(self, x, edge_index):
        # Initial Compression
        x = self.input_norm(x)
        x = self.pre_proj(x)
        x_in = x
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Skip connection
        x = x + self.skip1(x_in)
        
        # Layer 2 (Residual)
        h = self.conv2(x, edge_index)
        h = self.bn2(h)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = h + x # Full Residual connection
        
        return x

    def decode(self, z, src, dst):
        """
        Predict links between pairs of nodes given their embeddings.
        """
        h_u, h_v = z[src], z[dst]
        
        # Bilinear Interaction
        bilinear_out = self.bilinear(h_u, h_v)
        
        # Feature Engineering: [u, v, |u-v|, u*v]
        diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        
        # Combine all features including the Bilinear score
        pair_repr = torch.cat([h_u, h_v, diff, hadamard, bilinear_out], dim=1)
        return self.classifier(pair_repr)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        src, dst = edge_label_index
        return self.decode(z, src, dst)


class GINLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 3, dropout: float = 0.5):
        """
        Research-Grade GIN (Graph Isomorphism Network) for Link Prediction.
        GIN is theoretically more powerful at distinguishing structural motifs.
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        
        self.pre_proj = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        from torch_geometric.nn import GINConv
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # GIN layers use an MLP internally
        for i in range(num_layers):
            cur_in = 128 if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(cur_in, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(GNNBatchNorm(hidden_channels))
        
        self.dropout = dropout
        
        # Bilinear & Classifier heads
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        classifier_input_dim = (hidden_channels * 4) + 1
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def encode(self, x, edge_index):
        x = self.input_norm(x)
        x = self.pre_proj(x)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
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

# Backwards compatibility alias
GATLinkPredictor = GNNLinkPredictor
