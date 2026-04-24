# Achieve 95% Accuracy Pipeline (Presentation Fast-Track)
# This script automates the full training process for TransGraph-PPI using High-Performance Upgrades.
# Optimized for CPU training with 95%+ target.

$ErrorActionPreference = "Continue"

Write-Host "--- Step 1: Regenerating Topological Intelligence ---" -ForegroundColor Cyan
python -u src/data/graph_construction.py

Write-Host "--- Step 2: Training Ultra-High Capacity Sequence Model ---" -ForegroundColor Cyan
# Using 40 epochs for the deeper 1024-dim architecture
python -u src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt --epochs 40 --batch_size 128 --lr 0.001

Write-Host "--- Step 3: Training Neighborhood-Aware Graph Model (GraphSAGE) ---" -ForegroundColor Cyan
# SAGE is more robust for link prediction on large graphs
python -u src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt --model_type GAT --epochs 30 --lr 0.0005

Write-Host "--- Step 4: Training Deep Stacking Ensemble (XGBoost Meta-Learner) ---" -ForegroundColor Cyan
python -u src/training/train_ensemble.py

python -u src/analysis/compare_models.py
