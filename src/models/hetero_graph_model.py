import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, BatchNorm as GNNBatchNorm
from typing import Dict, List, Tuple

class HeteroGNNLinkPredictor(nn.Module):
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 in_channels_dict: Dict[str, int], 
                 hidden_channels: int = 128, 
                 dropout: float = 0.3):
        """
        Heterogeneous GNN Link Predictor for predicting protein-protein interactions.
        Integrates drug, disease, and pathway neighborhood context.
        """
        super().__init__()
        self.dropout = dropout
        
        # 1. Project node features of different dimensions to hidden_channels
        self.projs = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.projs[node_type] = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU()
            )
            
        # 2. Heterogeneous Convolution Layer 1
        conv1_relations = {}
        for edge_type in metadata[1]:
            src, rel, dst = edge_type
            # We use SAGEConv for aggregation
            conv1_relations[edge_type] = SAGEConv((-1, -1), hidden_channels)
            
        self.conv1 = HeteroConv(conv1_relations, aggr='mean')
        
        # Layer Normalization for each node type
        self.bns1 = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_channels) for ntype in metadata[0]
        })
        
        # 3. Heterogeneous Convolution Layer 2
        conv2_relations = {}
        for edge_type in metadata[1]:
            conv2_relations[edge_type] = SAGEConv((-1, -1), hidden_channels)
            
        self.conv2 = HeteroConv(conv2_relations, aggr='mean')
        
        self.bns2 = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_channels) for ntype in metadata[0]
        })
        
        # 4. Bilinear decoder for protein-protein link prediction
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        
        # Classifier input features: [u, v, |u - v|, u * v, bilinear]
        classifier_input_dim = (hidden_channels * 4) + 1
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def encode(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Project features
        h_dict = {ntype: self.projs[ntype](x) for ntype, x in x_dict.items()}
        
        # Layer 1
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {ntype: self.bns1[ntype](F.relu(h)) for ntype, h in h_dict.items()}
        h_dict = {ntype: F.dropout(h, p=self.dropout, training=self.training) for ntype, h in h_dict.items()}
        
        # Layer 2
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {ntype: self.bns2[ntype](F.relu(h)) for ntype, h in h_dict.items()}
        h_dict = {ntype: F.dropout(h, p=self.dropout, training=self.training) for ntype, h in h_dict.items()}
        
        return h_dict
        
    def decode(self, z_protein: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        h_u, h_v = z_protein[src], z_protein[dst]
        
        bilinear_out = self.bilinear(h_u, h_v)
        diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        
        pair_repr = torch.cat([h_u, h_v, diff, hadamard, bilinear_out], dim=1)
        return self.classifier(pair_repr)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], 
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(z_dict['protein'], edge_label_index)
