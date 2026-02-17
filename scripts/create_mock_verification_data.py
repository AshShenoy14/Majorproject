import torch
import pandas as pd
import sys
import os
from torch_geometric.data import Data
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor

def create_mock_data(limit=20):
    print(f"Creating mock verification data for top {limit} interactions...")
    
    # 1. Read Data
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        print("Error: train.csv not found")
        return

    df = pd.read_csv(train_path).head(limit)
    proteins = sorted(list(set(df["protein1"].unique()) | set(df["protein2"].unique())))
    print(f"Found {len(proteins)} unique proteins.")
    
    # 2. Mock Embeddings
    print("Generating mock embeddings...")
    embedding_dim = 320 # ESM-2 dime usually 320 for small models
    embeddings = {p: torch.randn(embedding_dim) for p in proteins}
    torch.save(embeddings, PROCESSED_DATA_DIR / "embeddings.pt")
    
    # 3. Mock Graph
    print("Generating mock graph...")
    node_mapping = {p: i for i, p in enumerate(proteins)}
    
    src = []
    dst = []
    for _, row in df.iterrows():
        if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
            u, v = node_mapping[row["protein1"]], node_mapping[row["protein2"]]
            src.append(u)
            dst.append(v)
            src.append(v)
            dst.append(u)
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.stack([embeddings[p] for p in proteins])
    
    data = Data(x=x, edge_index=edge_index)
    
    torch.save(data, PROCESSED_DATA_DIR / "ppi_graph.pt")
    torch.save(node_mapping, PROCESSED_DATA_DIR / "ppi_graph_mapping.pt")
    
    # 4. Mock Models
    print("Saving mock model weights...")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Seq Model
    seq_model = SequencePPIModel(input_dim=embedding_dim)
    torch.save(seq_model.state_dict(), models_dir / "sequence_model_best.pth")
    
    # Graph Model
    graph_model = GATLinkPredictor(in_channels=embedding_dim, hidden_channels=64)
    torch.save(graph_model.state_dict(), models_dir / "graph_model_best.pth")
    
    # Ensemble (Optional, can skip if backend handles missing)
    # We won't save ensemble pickle, backend should fallback to soft voting if missing.
    
    print("Mock data created successfully.")

if __name__ == "__main__":
    create_mock_data()
