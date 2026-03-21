# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Union

from src.utils.bio_encoder import BioFeatureEncoder

class PPIDataset(Dataset):
    def __init__(self, data: Union[str, pd.DataFrame], embeddings: Dict[str, torch.Tensor], bio_mapping: Dict[str, torch.Tensor] = None):
        """
        Args:
            data: Path to the csv file OR a pandas DataFrame (protein1, protein2, label).
            embeddings: Dictionary mapping ProteinID -> Embedding.
            bio_mapping: Dictionary mapping ProteinID -> Biological Feature Vector.
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.read_csv(data)
            
        self.embeddings = embeddings
        self.bio_mapping = bio_mapping or {}
        self.bio_dim = 0
        if self.bio_mapping:
            first_val = next(iter(self.bio_mapping.values()))
            self.bio_dim = first_val.shape[0]

        # Filter out pairs where embeddings are missing
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
        
        # Convert to float32 on-the-fly
        emb1 = self.embeddings[p1].float()
        emb2 = self.embeddings[p2].float()
        
        # Mean-pool per-residue embeddings
        if emb1.dim() > 1:
            emb1 = emb1.mean(dim=0)
        if emb2.dim() > 1:
            emb2 = emb2.mean(dim=0)
            
        # Append Bio-Features if available
        if self.bio_mapping:
            bio1 = self.bio_mapping.get(p1, torch.zeros(self.bio_dim))
            bio2 = self.bio_mapping.get(p2, torch.zeros(self.bio_dim))
            emb1 = torch.cat([emb1, bio1])
            emb2 = torch.cat([emb2, bio2])
        
        return emb1, emb2, torch.tensor(label, dtype=torch.float32)
