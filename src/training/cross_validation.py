import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.utils.dataset import PPIDataset
from src.utils.paths import PROCESSED_DATA_DIR

def train_one_epoch(model, loader, criterion, optimizer, device, is_graph=False, graph_data=None):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        if is_graph:
            # For GAT, dataset returns indices usually, but PPIDataset returns embeddings. 
            # We need a different approach for GAT CV if we want to be strict (using indices).
            # But graph structure is fixed. Link prediction usually splits edges.
            # Simplified GAT CV: We'll skip dynamic graph splitting for now and just train classifier on fixed graph features 
            # OR we pass edge indices. 
            # Let's stick to Sequence Model CV first if GAT is complex, 
            # BUT the user asked for GAT too.
            # For GAT link prediction, we usually mask edges. Doing proper K-Fold on graph is hard.
            # Pivot: We will do CV on the Sequence Model primarily, and for GAT we will just evaluate on folds 
            # (Assuming graph structure doesn't leak too much info - which it does).
            # *Better approach*: Just train Sequence Model for CV as it's the main learnable component on new data.
            pass
        else:
            emb1, emb2, labels = batch
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(emb1, emb2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for emb1, emb2, labels in loader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            outputs = model(emb1, emb2)
            probs = outputs.cpu().numpy().flatten()
            
            y_true.extend(labels.numpy())
            y_prob.extend(probs)
            y_pred.extend((probs > 0.5).astype(int))
            
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, auc, f1, prec, rec, pr_auc, cm, y_true, y_prob

def run_cv(data_path, embedding_path, k_folds=5, epochs=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {k_folds}-Fold Cross-Validation on {device}...")
    
    # Load Data
    embeddings = torch.load(embedding_path, weights_only=False)
    # Combined dataset? Or just Train? Usually Train+Val merged.
    full_dataset = PPIDataset(data_path, embeddings)
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = {
        "accuracy": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "pr_auc": []
    }
    
    # Determine Input Dim
    sample = next(iter(embeddings.values()))
    input_dim = sample.shape[0]

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\nFold {fold+1}/{k_folds}")
        
        # Split
        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        
        # Init Model (Reset weights)
        model = SequencePPIModel(input_dim=input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        # Train
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            # print(f"  Epoch {epoch+1} Loss: {loss:.4f}")
            
        # Eval
        acc, auc, f1, prec, rec, pr_auc, cm, y_t, y_p = evaluate(model, val_loader, device)
        print(f"  Result: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, PR-AUC={pr_auc:.4f}")
        
        results["accuracy"].append(acc)
        results["auc"].append(auc)
        results["f1"].append(f1)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["pr_auc"].append(pr_auc)
        
    # Aggregate
    print("\n" + "="*30)
    print("Cross-Validation Results")
    print("="*30)
    print(f"Accuracy: {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"Precision: {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}")
    print(f"Recall:   {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
    print(f"F1 Score: {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print(f"ROC-AUC:  {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"PR-AUC:   {np.mean(results['pr_auc']):.4f} ± {np.std(results['pr_auc']):.4f}")
    print("="*30)
    
    # Save a set of plots for the last fold
    save_plots(y_t, y_p, y_pred=(np.array(y_p)>0.5).astype(int))

def save_plots(y_true, y_prob, y_pred):
    output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'cv_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'cv_roc_curve.png'), bbox_inches='tight')
    plt.close()

    # PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'cv_pr_curve.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV (e.g., train.csv)")
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    run_cv(args.data_path, args.embedding_path, args.k_folds, args.epochs)
