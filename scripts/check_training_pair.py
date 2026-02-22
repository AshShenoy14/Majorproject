import pandas as pd
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.paths import PROCESSED_DATA_DIR

def check_pair(p1, p2):
    print(f"Checking pair: {p1} - {p2}")
    
    found = False
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = PROCESSED_DATA_DIR / split
        if path.exists():
            df = pd.read_csv(path)
            # Check both directions
            mask = ((df["protein1"] == p1) & (df["protein2"] == p2)) | \
                   ((df["protein1"] == p2) & (df["protein2"] == p1))
            
            if mask.any():
                print(f"Found in {split}!")
                print(df[mask])
                found = True
    
    if not found:
        print("Pair NOT found in any split.")

if __name__ == "__main__":
    p1 = "ENSP00000269305"
    p2 = "ENSP00000361423"
    check_pair(p1, p2)
