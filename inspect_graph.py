import torch
from torch_geometric.data import Data

data = torch.load(r'e:\majorproject\data\processed\ppi_graph.pt', weights_only=False)
print(f"Nodes: {data.x.shape[0]}")
print(f"Features: {data.x.shape[1]}")
print(f"Edges: {data.edge_index.shape[1]}")
