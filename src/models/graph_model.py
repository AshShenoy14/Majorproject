import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Output of conv1 is hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
        # Link Prediction Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index, edge_label_index):
        """
        Args:
            x: Node features (batch_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_label_index: Pairs to predict (2, num_pairs)
        """
        z = self.encode(x, edge_index)
        
        src, dst = edge_label_index
        # Concatenate embeddings of the pair
        emb_pair = torch.cat([z[src], z[dst]], dim=1)
        
        return self.classifier(emb_pair)
