import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, Any

def build_rf_features_for_df(
    df: pd.DataFrame,
    embeddings: Dict[str, torch.Tensor],
    bio_mapping: Dict[str, torch.Tensor],
    bio_manager: Any,
    desc: str = "Extracting RF Features"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Constructs deterministic pair features for Random Forest:
    [e1, e2, abs(e1 - e2), e1 * e2, b1, b2, BioScore]
    Dimension: 480 + 480 + 480 + 480 + 10 + 10 + 1 = 1941 features.
    
    Returns:
        X: numpy array of shape (N, 1941)
        y: numpy array of shape (N,)
        valid_df: filtered dataframe with valid samples
    """
    valid_mask = (
        df["protein1"].isin(embeddings) & 
        df["protein2"].isin(embeddings)
    )
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    
    missing_count = len(df) - len(valid_df)
    if missing_count > 0:
        print(f"Warning: skipped {missing_count} pairs due to missing embeddings.")
        
    X_list = []
    y_list = []
    
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc=desc):
        p1, p2 = row["protein1"], row["protein2"]
        label = row["label"]
        
        # 1 & 2. ESM embeddings (480-dim each)
        e1_t = embeddings[p1].float()
        e2_t = embeddings[p2].float()
        e1 = e1_t.mean(dim=0).cpu().numpy() if e1_t.dim() > 1 else e1_t.cpu().numpy()
        e2 = e2_t.mean(dim=0).cpu().numpy() if e2_t.dim() > 1 else e2_t.cpu().numpy()
        
        # 3. Absolute difference |e1 - e2| (480-dim)
        diff = np.abs(e1 - e2)
        
        # 4. Element-wise product e1 * e2 (480-dim)
        prod = e1 * e2
        
        # 5 & 6. Bio localization multi-hot encodings (10-dim each)
        b1_t = bio_mapping.get(p1, torch.zeros(10))
        b2_t = bio_mapping.get(p2, torch.zeros(10))
        b1 = b1_t.cpu().numpy() if isinstance(b1_t, torch.Tensor) else np.array(b1_t)
        b2 = b2_t.cpu().numpy() if isinstance(b2_t, torch.Tensor) else np.array(b2_t)
        
        # 7. Biological compatibility score (1-dim)
        comp = bio_manager.check_localization_compatibility(p1, p2, fetch_missing=False)
        bio_score = comp.get("score", 0.5)
        
        # Feature vector concatenation
        feat = np.concatenate([e1, e2, diff, prod, b1, b2, [bio_score]])
        X_list.append(feat)
        y_list.append(label)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    assert X.shape[1] == 1941, f"Expected 1941 features, but constructed {X.shape[1]}"
    assert not np.isnan(X).any(), "NaN values found in feature matrix X!"
    assert not np.isinf(X).any(), "Inf values found in feature matrix X!"
    
    return X, y, valid_df
