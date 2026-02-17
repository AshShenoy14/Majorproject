import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.utils.dataset import PPIDataset
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def train(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, embedding_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Sequence Model on {device}...")
    
    # Load embeddings (Mocking for now if file logic not fully ready)
    # In real run: embeddings = torch.load(embedding_path)
    # For now, we assume embeddings are passed or loaded from a standard location
    # If embedding_path is not provided, we might fail or generate dummy
    
    if embedding_path and os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, weights_only=False)
    else:
        print("No embedding file provided/found. Cannot proceed without embeddings.")
        # For demonstration context, we might generate random embeddings if needed, but better to fail.
        return

    # Datasets
    train_dataset = PPIDataset(PROCESSED_DATA_DIR / "train.csv", embeddings)
    val_dataset = PPIDataset(PROCESSED_DATA_DIR / "val.csv", embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    # Input dim from embeddings
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[0]
    
    model = SequencePPIModel(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for emb1, emb2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(emb1, emb2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
                outputs = model(emb1, emb2)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "sequence_model_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to dictionary of protein embeddings (.pt)")
    args = parser.parse_args()
    
    train(args.epochs, args.batch_size, args.lr, args.embedding_path)
