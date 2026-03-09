import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        # Output of conv1 is hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        
        # Link Prediction Classifier
        # Input: h_u, h_v, |h_u - h_v|, h_u * h_v → hidden_channels × 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 1)
            # No Sigmoid — use BCEWithLogitsLoss during training
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

    def forward(self, x, edge_index, edge_label_index):
        """
        Args:
            x: Node features (batch_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_label_index: Pairs to predict (2, num_pairs)
        Returns:
            Raw logits (no sigmoid). Apply sigmoid yourself during inference.
        """
        z = self.encode(x, edge_index)
        
        src, dst = edge_label_index
        h_u = z[src]
        h_v = z[dst]
        
        abs_diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        
        emb_pair = torch.cat([h_u, h_v, abs_diff, hadamard], dim=1)
        
        return self.classifier(emb_pair)
