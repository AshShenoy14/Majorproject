# Achieve 95% Accuracy Pipeline (Presentation Fast-Track)
# This script automates the full training process for TransGraph-PPI using High-Performance Upgrades.
# Optimized for CPU training with 95%+ target.

$ErrorActionPreference = "Continue"

Write-Host "--- Step 1: Regenerating Topological Intelligence ---" -ForegroundColor Cyan
python -u src/data/graph_construction.py

# Clear old logs and metrics
if (Test-Path "presentation_metrics.txt") { Remove-Item "presentation_metrics.txt" }

Write-Host "--- Step 2: Training Hyper-Optimized Sequence Model (Symmetric + Stable) ---" -ForegroundColor Cyan
# 1e-3 is a stable starting point for AdamW on CPU
python -u src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt --epochs 25 --batch_size 128 --lr 0.001

Write-Host "--- Step 3: Training Focal-Loss Graph Model (GIN + Structural) ---" -ForegroundColor Cyan
# GIN with focal loss focuses on the hard structural motifs
python -u src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt --model_type GIN --epochs 30 --lr 0.001

Write-Host "--- Step 4: Training Deep Stacking Ensemble (XGBoost Meta-Learner) ---" -ForegroundColor Cyan
python -u src/training/train_ensemble.py

Write-Host "--- Step 5: Final Presentation Audit (95%+ Target Verification) ---" -ForegroundColor Cyan
# This script validates the model and generates the final 'Wow' metrics file
python -u scripts/presentation_audit.py
