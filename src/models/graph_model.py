import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        # Refined capacity for stability
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_channels * heads)
        
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_channels * heads)
        
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.ln3 = nn.LayerNorm(hidden_channels)
        
        self.res_proj1 = nn.Linear(in_channels, hidden_channels * heads)
        
        # Skip connection for raw signal
        self.skip_proj1 = nn.Linear(in_channels, hidden_channels)
        
        self.dropout = dropout
        
        # Aligned Link Prediction Classifier
        # Input: h_u, h_v, |h_u - h_v|, h_u * h_v → (hidden_channels) × 4
        pair_dim = hidden_channels * 4
        classifier_hidden = 512
        
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.BatchNorm1d(classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(classifier_hidden // 2, 1)
        )

    def encode(self, x, edge_index):
        # Layer 1
        identity = x
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x + self.res_proj1(identity)
        
        # Layer 2
        identity = x
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x + identity
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.ln3(x)
        x = F.elu(x)
        
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
        
        # Project raw features for skip connection
        z_raw = self.skip_proj1(x)
        
        src, dst = edge_label_index
        
        # Combine GAT learned features with raw projected features
        h_u = z[src] + z_raw[src]
        h_v = z[dst] + z_raw[dst]
        
        abs_diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        
        emb_pair = torch.cat([h_u, h_v, abs_diff, hadamard], dim=1)
        
        return self.classifier(emb_pair)
