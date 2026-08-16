import argparse
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.utils.bio_encoder import BioFeatureEncoder
from src.analysis.biological_managers import BiologicalManager
from src.utils.rf_feature_builder import build_rf_features_for_df

def find_optimal_threshold(y_true, y_prob):
    """
    Sweeps thresholds from 0.10 to 0.90 to find the F1-optimal threshold on validation data.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob > thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    return best_thresh, best_f1

def train_random_forest(
    n_estimators: int = 100,
    max_depth: int = None,
    n_jobs: int = -1,
    random_state: int = 42,
    dry_run: bool = False,
    max_samples: int = 500
):
    print("=" * 70)
    print("      TRANSGRAPH-PPI: TRADITIONAL RANDOM FOREST BASELINE TRAINING      ")
    print("=" * 70)
    print(f"Configuration: n_estimators={n_estimators}, max_depth={max_depth}, dry_run={dry_run}")
    
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_path}")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Required train.csv or val.csv missing from processed data.")
        
    print("\n[1/5] Loading ESM Embeddings & Biological Utilities...")
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = {k: v.float() if v.dtype == torch.float16 else v for k, v in embeddings.items()}
    print(f"  Loaded {len(embeddings)} protein ESM embeddings.")
    
    bio_encoder = BioFeatureEncoder()
    bio_mapping = bio_encoder.get_feature_map()
    print(f"  Loaded biological localization multi-hot encodings for {len(bio_mapping)} proteins.")
    
    bio_manager = BiologicalManager()
    print("  Initialized BiologicalManager for localization compatibility scoring.")
    
    # Load Training Data
    print("\n[2/5] Constructing Feature Matrix for Training Data (train.csv ONLY)...")
    train_df = pd.read_csv(train_path)
    if dry_run:
        print(f"  [DRY-RUN] Subsampling train.csv to first {max_samples} samples...")
        train_df = train_df.iloc[:max_samples].copy().reset_index(drop=True)
        
    X_train, y_train, filtered_train_df = build_rf_features_for_df(
        train_df, embeddings, bio_mapping, bio_manager, desc="Extracting Train Features"
    )
    
    print(f"\n[LEAKAGE CHECK & FEATURE AUDIT]")
    print(f"  Training Samples: {len(X_train)} (Positive: {sum(y_train==1)}, Negative: {sum(y_train==0)})")
    print(f"  Feature Vector Shape: {X_train.shape} (Exact Feature Dimension: {X_train.shape[1]})")
    print(f"  Missing Embedding Pairs Filtered: {len(train_df) - len(filtered_train_df)}")
    print(f"  Zero NaNs: {not np.isnan(X_train).any()}, Zero Infs: {not np.isinf(X_train).any()}")
    print("  [VERIFIED] Fitted strictly on train.csv. val.csv and test.csv were NOT touched during fitting.")
    
    # Train Random Forest Classifier
    print(f"\n[3/5] Fitting RandomForestClassifier ({n_estimators} trees, n_jobs={n_jobs})...")
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1 if not dry_run else 0
    )
    rf_clf.fit(X_train, y_train)
    print("  Model training complete.")
    
    # Save Model Checkpoint
    save_path = PROJECT_ROOT / "models" / "random_forest_baseline.pkl"
    os.makedirs(save_path.parent, exist_ok=True)
    joblib.dump(rf_clf, save_path)
    print(f"  [SAVED] Saved Random Forest model checkpoint to: {save_path}")
    
    # Evaluate on Validation Data for F1 Threshold Optimization
    print("\n[4/5] Evaluating on Validation Data (val.csv) for Threshold Tuning...")
    val_df = pd.read_csv(val_path)
    if dry_run:
        val_df = val_df.iloc[:200].copy().reset_index(drop=True)
        
    X_val, y_val, filtered_val_df = build_rf_features_for_df(
        val_df, embeddings, bio_mapping, bio_manager, desc="Extracting Val Features"
    )
    
    val_probs = rf_clf.predict_proba(X_val)[:, 1]
    
    # Optimal Threshold Sweep on val.csv
    val_thresh, best_val_f1 = find_optimal_threshold(y_val, val_probs)
    
    val_preds_default = (val_probs > 0.5).astype(int)
    val_preds_opt = (val_probs > val_thresh).astype(int)
    
    acc_def = accuracy_score(y_val, val_preds_default)
    prec_def = precision_score(y_val, val_preds_default, zero_division=0)
    rec_def = recall_score(y_val, val_preds_default, zero_division=0)
    f1_def = f1_score(y_val, val_preds_default, zero_division=0)
    
    acc_opt = accuracy_score(y_val, val_preds_opt)
    prec_opt = precision_score(y_val, val_preds_opt, zero_division=0)
    rec_opt = recall_score(y_val, val_preds_opt, zero_division=0)
    f1_opt = f1_score(y_val, val_preds_opt, zero_division=0)
    
    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    
    print("\n[5/5] VALIDATION PERFORMANCE SUMMARY (val.csv ONLY)")
    print("-" * 65)
    print(f"  Validation Samples evaluated: {len(X_val)}")
    print(f"  ROC-AUC Score: {roc_auc:.4f}")
    print(f"  PR-AUC Score:  {pr_auc:.4f}")
    print(f"  Default Threshold (0.50): Acc={acc_def:.4f}, Prec={prec_def:.4f}, Rec={rec_def:.4f}, F1={f1_def:.4f}")
    print(f"  Optimal Threshold ({val_thresh:.4f}): Acc={acc_opt:.4f}, Prec={prec_opt:.4f}, Rec={rec_opt:.4f}, F1={f1_opt:.4f}")
    print("-" * 65)
    print(f"\nClassification Report at Optimal Threshold ({val_thresh:.4f}):")
    print(classification_report(y_val, val_preds_opt, digits=4))
    
    print("\n[SUMMARY & LEAKAGE ASSERTION]")
    print(f"  1. Model fit: ONLY on train.csv ({len(X_train)} samples)")
    print(f"  2. Feature vector: 1941 deterministic biophysical & biological features")
    print(f"  3. Threshold selected: {val_thresh:.4f} strictly on val.csv")
    print(f"  4. test.csv status: UNTOUCHED during this training run")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traditional Random Forest Baseline for PPI Prediction.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of trees")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Run fast dry-run verification on a small subset")
    parser.add_argument("--max_samples", type=int, default=500, help="Max train samples for dry-run")
    args = parser.parse_args()

    train_random_forest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        dry_run=args.dry_run,
        max_samples=args.max_samples
    )
