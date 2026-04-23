# Achieve 95% Accuracy Pipeline (Optimized)
# This script automates the full training process for TransGraph-PPI using High-Performance Upgrades.

Write-Host "--- Step 1: Clearing Stale Checkpoints & Models ---" -ForegroundColor Cyan
Remove-Item checkpoints/*.pt -ErrorAction SilentlyContinue
Remove-Item models/*.pth -ErrorAction SilentlyContinue

Write-Host "--- Step 2: Training Augmented Sequence Model (Symmetric + Noise) ---" -ForegroundColor Cyan
# Running 20 epochs for a quick but powerful boost
python -u src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt --epochs 20 --batch_size 128

Write-Host "--- Step 3: Training Focal-Loss Graph Model (GIN + Focal) ---" -ForegroundColor Cyan
# GIN is more structurally powerful than GAT
python -u src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt --model_type GIN --epochs 20

Write-Host "--- Step 4: Training Deep Stacking Ensemble ---" -ForegroundColor Cyan
python -u src/training/train_ensemble.py

Write-Host "--- Step 5: Final Presentation Audit (95%+ Target) ---" -ForegroundColor Cyan
python -u scripts/presentation_audit.py

Write-Host "--- Pipeline Complete! See presentation_metrics.txt for the 'Wow' results ---" -ForegroundColor Green
