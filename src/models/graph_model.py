import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, LayerNorm as GNNLayerNorm

class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, heads: int = 4, dropout: float = 0.4):
        """
        Refined Hybrid GNN Link Predictor.
        Increased capacity to 128 channels for better ESM feature preservation.
        """
        super().__init__()
        
        # 1. Feature Encoding Layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.ln1 = GNNLayerNorm(hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.ln2 = GNNLayerNorm(hidden_channels)
        
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.ln3 = GNNLayerNorm(hidden_channels)
        
        # Strong Residual Skips
        self.res_proj = nn.Linear(in_channels, hidden_channels)
        
        self.dropout = dropout
        
        # 2. Link Prediction Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(256, 1)
        )

    def encode(self, x, edge_index):
        # Layer 1: Global Context
        identity = self.res_proj(x)
        h = self.conv1(x, edge_index)
        h = self.ln1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + identity
        
        # Layer 2: Attention based Refinement
        identity = h
        h = self.conv2(h, edge_index)
        h = self.ln2(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + identity # Residual
        
        # Layer 3: Aggregation
        h = self.conv3(h, edge_index)
        h = self.ln3(h)
        h = F.gelu(h)
        
        return h

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        
        src, dst = edge_label_index
        h_u, h_v = z[src], z[dst]
        
        # Feature engineering for pairs
        abs_diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        
        pair_repr = torch.cat([h_u, h_v, abs_diff, hadamard], dim=1)
        return self.classifier(pair_repr)
