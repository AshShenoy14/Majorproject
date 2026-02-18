import sys
import os
import pandas as pd
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

from src.data.target_manager import TargetManager
from src.utils.paths import PROCESSED_DATA_DIR

def populate():
    print("Initializing Target Manager...")
    manager = TargetManager()
    
    # Load some proteins from train.csv to populate for
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        print("Train.csv not found.")
        return

    print("Reading train.csv...")
    df = pd.read_csv(train_path)
    
    # Get top proteins by frequency (degree)
    all_proteins = pd.concat([df['protein1'], df['protein2']])
    top_proteins = all_proteins.value_counts().head(50).index.tolist()
    
    print(f"Fetching targets for top {len(top_proteins)} proteins...")
    print(top_proteins[:5])
    
    targets = manager.get_targets(top_proteins)
    
    print(f"Done. Found {len(targets)} targets.")
    if not targets.empty:
        print(targets.head())

if __name__ == "__main__":
    populate()
