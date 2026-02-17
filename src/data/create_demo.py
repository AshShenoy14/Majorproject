import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR

def create_demo_data(num_samples=100):
   print(f"Creating demo dataset with {num_samples} samples...")
   
   # Create mock proteins
   proteins = [f"P{i}" for i in range(num_samples)]
   
   # Create mock interactions
   data = []
   for i in range(num_samples * 2):
       p1 = np.random.choice(proteins)
       p2 = np.random.choice(proteins)
       label = np.random.randint(0, 2)
       data.append({"protein1": p1, "protein2": p2, "label": label})
       
   df = pd.DataFrame(data)
   
   # Split
   train = df.sample(frac=0.8)
   rest = df.drop(train.index)
   val = rest.sample(frac=0.5)
   test = rest.drop(val.index)
   
   train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
   val.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
   test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
   print("Demo data created.")

if __name__ == "__main__":
   create_demo_data()
