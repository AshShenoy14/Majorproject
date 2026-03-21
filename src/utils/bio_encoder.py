import numpy as np
# -*- coding: utf-8 -*-
import pandas as pd
import torch
from typing import List, Dict
import os
from src.utils.paths import PROCESSED_DATA_DIR

class BioFeatureEncoder:
    def __init__(self, cache_file: str = "bio_metadata_cache.csv"):
        self.cache_path = PROCESSED_DATA_DIR / cache_file
        self.locations = [
            "nucleus", "cytoplasm", "cell membrane", "mitochondrion", 
            "endoplasmic reticulum", "golgi apparatus", "lysosome", 
            "peroxisome", "secreted", "extracellular"
        ]
        self.loc_to_idx = {loc: i for i, loc in enumerate(self.locations)}
        self.dim = len(self.locations)

    def encode_protein(self, localization_str: str) -> torch.Tensor:
        """
        Encodes a localization string into a multi-hot vector.
        """
        vec = torch.zeros(self.dim)
        if not localization_str or pd.isna(localization_str):
            return vec
            
        locs = [l.strip().lower() for l in str(localization_str).split(";")]
        for l in locs:
            for known_loc in self.locations:
                if known_loc in l:
                    vec[self.loc_to_idx[known_loc]] = 1.0
        return vec

    def get_feature_map(self) -> Dict[str, torch.Tensor]:
        """
        Returns a mapping of ProteinID -> Biological Feature Vector.
        """
        if not self.cache_path.exists():
            return {}
            
        df = pd.read_csv(self.cache_path)
        feature_map = {}
        for _, row in df.iterrows():
            pid = row["protein_id"]
            loc_str = row.get("localization", "")
            feature_map[pid] = self.encode_protein(loc_str)
        return feature_map
