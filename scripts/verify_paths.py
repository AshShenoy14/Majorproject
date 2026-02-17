import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to sys.path, mimicking main.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts/
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

print(f"Project Root: {project_root}")

try:
    from src.utils.paths import PROCESSED_DATA_DIR
    print(f"Imported PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")

    train_path = PROCESSED_DATA_DIR / "train.csv"
    print(f"Looking for train.csv at: {train_path}")
    
    if train_path.exists():
        print("SUCCESS: train.csv found.")
        try:
            df = pd.read_csv(train_path)
            print(f"SUCCESS: Read {len(df)} rows from train.csv.")
            print(f"Columns: {df.columns.tolist()}")
            
            positive = df[df["label"] == 1].head(20)
            print(f"Found {len(positive)} positive interactions in first 20 rows (if limit applied).")
            
        except Exception as e:
            print(f"ERROR: Failed to read csv: {e}")
    else:
        print("ERROR: train.csv NOT found.")

except ImportError as e:
    print(f"ERROR: Could not import from src: {e}")
except Exception as e:
    print(f"ERROR: Unexpected error: {e}")
