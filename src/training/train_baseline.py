import argparse
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def load_data_for_rf(data_path, embeddings):
    """
    Loads data and creates feature vectors for Random Forest.
    Features: Concatenation of Protein 1 and Protein 2 embeddings.
    """
    df = pd.read_csv(data_path)
    X = []
    y = []
    
    print(f"Loading data from {data_path}...")
    valid_count = 0
    missing_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        
        if p1 in embeddings and p2 in embeddings:
            e1 = embeddings[p1].numpy()
            e2 = embeddings[p2].numpy()
            # Feature: Concatenate embeddings (N, dim*2)
            # You could also try element-wise product or difference
            feature = np.concatenate([e1, e2]) 
            X.append(feature)
            y.append(label)
            valid_count += 1
        else:
            missing_count += 1
            
    if missing_count > 0:
        print(f"Warning: skipped {missing_count} pairs due to missing embeddings.")
        
    return np.array(X), np.array(y)

def train_rf(embedding_path: str, n_estimators: int = 100):
    print("Loading Embeddings...")
    if not os.path.exists(embedding_path):
        print("Error: Embedding file not found.")
        return

    embeddings = torch.load(embedding_path, weights_only=False)
    
    # Load Train Data
    X_train, y_train = load_data_for_rf(PROCESSED_DATA_DIR / "train.csv", embeddings)
    
    # Load Validation Data
    X_val, y_val = load_data_for_rf(PROCESSED_DATA_DIR / "val.csv", embeddings)
    
    print(f"Training Random Forest with {n_estimators} trees...")
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating on Validation Set...")
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_prob)
    
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC-AUC: {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save Model
    save_path = PROJECT_ROOT / "models" / "baseline_rf.pkl"
    os.makedirs(save_path.parent, exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"Baseline model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to embeddings (.pt)")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    
    train_rf(args.embedding_path, args.n_estimators)
