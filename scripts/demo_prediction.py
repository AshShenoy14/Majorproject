import torch
import sys
import os
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sequence_model import SequencePPIModel
from src.utils.paths import PROJECT_ROOT, PROCESSED_DATA_DIR

def predict():
    print("=== TransGraph-PPI Prediction Demo ===")
    
    # 1. Load Embeddings
    print("Loading embeddings...")
    # Try temp file first (from recent run), then main file
    emb_path = PROCESSED_DATA_DIR / "temp_embeddings.pt"
    if not emb_path.exists():
        emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
        
    if not emb_path.exists():
        print("Error: No embeddings file found. Run pipeline first.")
        return

    try:
        embeddings = torch.load(emb_path, weights_only=False)
        print(f"Loaded {len(embeddings)} protein embeddings.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # 2. Load Model
    print("Loading Sequence Model...")
    model_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get input dim from first embedding
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[0]
    
    model = SequencePPIModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Interactive Loop
    while True:
        print("\n--- New Prediction ---")
        p1 = input("Enter Protein 1 ID (or 'q' to quit): ").strip()
        if p1.lower() == 'q':
            break
            
        p2 = input("Enter Protein 2 ID: ").strip()
        
        if p1 not in embeddings:
            print(f"Error: Protein {p1} not found in embeddings.")
            continue
        if p2 not in embeddings:
            print(f"Error: Protein {p2} not found in embeddings.")
            continue
            
        emb1 = embeddings[p1].unsqueeze(0).to(device)
        emb2 = embeddings[p2].unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(emb1, emb2).item()
            
        print(f"Interaction Probability: {score:.4f}")
        if score > 0.5:
            print(">> INTERACTION PREDICTED")
        else:
            print(">> NO INTERACTION PREDICTED")

if __name__ == "__main__":
    predict()
