# Achieve 95% Accuracy Pipeline (Optimized)
# This script automates the full training process for TransGraph-PPI using High-Performance Upgrades.

Write-Host "--- Step 1: Regenerating Enriched Graph (Topological Intelligence) ---" -ForegroundColor Cyan
python -u src/data/graph_construction.py
Remove-Item checkpoints/*.pt -ErrorAction SilentlyContinue
Remove-Item models/*.pth -ErrorAction SilentlyContinue

Write-Host "--- Step 2: Training Augmented Sequence Model (Symmetric + Noise) ---" -ForegroundColor Cyan
# Running 60 epochs for a deep sequence-based signal
python -u src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt --epochs 60 --batch_size 128

Write-Host "--- Step 3: Training Focal-Loss Graph Model (GIN + Topological) ---" -ForegroundColor Cyan
# GIN with 100 epochs to fully learn structural motifs
python -u src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt --model_type GIN --epochs 100

Write-Host "--- Step 4: Training Deep Stacking Ensemble ---" -ForegroundColor Cyan
python -u src/training/train_ensemble.py

Write-Host "--- Step 5: Final Presentation Audit (95%+ Target) ---" -ForegroundColor Cyan
python -u scripts/presentation_audit.py

Write-Host "--- Pipeline Complete! See presentation_metrics.txt for the 'Wow' results ---" -ForegroundColor Green
