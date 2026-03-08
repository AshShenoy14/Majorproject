import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict

class PPIDataset(Dataset):
    def __init__(self, data_path: str, embeddings: Dict[str, torch.Tensor]):
        """
        Args:
            data_path: Path to the csv file (protein1, protein2, label).
            embeddings: Dictionary mapping ProteinID -> Embedding.
        """
        self.df = pd.read_csv(data_path)
        self.embeddings = embeddings
        
        # Filter out pairs where embeddings are missing
        # In a real scenario, we should handle this earlier or ensure completeness.
        valid_indices = []
        for idx, row in self.df.iterrows():
            if row["protein1"] in self.embeddings and row["protein2"] in self.embeddings:
                valid_indices.append(idx)
        
        if len(valid_indices) < len(self.df):
            print(f"Warning: Dropped {len(self.df) - len(valid_indices)} pairs due to missing embeddings.")
            self.df = self.df.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p1 = row["protein1"]
        p2 = row["protein2"]
        label = row["label"]
        
        # Convert to float32 on-the-fly (embeddings may be stored as float16 to save RAM)
        emb1 = self.embeddings[p1].float()
        emb2 = self.embeddings[p2].float()
        
        return emb1, emb2, torch.tensor(label, dtype=torch.float32)
