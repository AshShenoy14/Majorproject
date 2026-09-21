# Colab run evidence (console excerpt)

> **Run 1 log.** This excerpt is from the first Colab run. The repository's current artifacts come from a second, independent Colab run (finished 2026-09-21 16:44 UTC), whose
> checkpoint hashes are in `assets/evaluation/final_test_metrics.json` and whose data/model hashes are in `assets/evaluation/artifact_hashes.txt`. Run 2 differs by <=0.1 point
> (ensemble accuracy 0.9213 vs 0.9211; GraphSAGE and ensemble checkpoints differ, the sequence model and random-forest checkpoints are byte-identical). No console log of run 2 is available;
> its fold-leak assertions are not separately evidenced, but the code path is unchanged and run 1's assertions fired.

Excerpt of the console output of the Colab T4 run that produced `models/*` and `assets/evaluation/final_test_metrics.json`
(hash prefixes of those checkpoints are recorded in the JSON). Only the lines relevant to verification are kept; the
per-epoch training lines are omitted. This is a transcription of pasted console output, not a machine-generated log file.

## Data (preprocess_data.py + scripts/verify_splits.py, on Colab)
```
Negative contamination check passed: 0 negatives in train/val/test appear in the STRING any-confidence (score > 0) interaction set.
Train: 161369 rows (Pos: 80685, Neg: 80684) | Val: 20171 (10085/10086) | Test: 20172 (10086/10086)
Train AND Validation: 0 | Train AND Test: 0 | Validation AND Test: 0   (canonical pairs)
Train Unique Proteins: 12323 | Val: 10231 | Test: 10278
Test Proteins in Train: 10278 / 10278 (100.0%) [Transductive Link Prediction]
degree_bias_baseline: test_accuracy 0.7063, test_roc_auc 0.7838
Enriched Graph: 12323 nodes, 161370 edges, 483 features.
```

## Fold-leak assertions (train_ensemble.py) - fired in the real run
```
Fold 1/5 ... [VERIFIED] no held-out / early-stop pair is an edge of the fold graph (58093 positive edges).
Fold 2/5 ... [VERIFIED] (58093 positive edges)   Fold 3/5 ... [VERIFIED] (58093)
Fold 4/5 ... [VERIFIED] (58093)                  Fold 5/5 ... [VERIFIED] (58093)
[VERIFIED] All 161369 training samples received exactly one leakage-free OOF prediction.
OOF GraphSAGE calibration  before: ECE 0.13337 Brier 0.11657  after: ECE 0.0622 Brier 0.08673
OOF AUC  seq 0.9331 | graph 0.9402
Training Deep Ensemble on 161369 samples with 7 features...
```
(The `assert not torch.equal(x_fold[:, -3:], base_x[:, -3:])` fold-feature assertion has no print statement; it passed because the run did not abort.)

## Calibration (train_graph_model.py, val.csv)
```
before: ECE 0.15287 Brier 0.11847 | after: ECE 0.05035 Brier 0.07456 | cross-fitted: ECE 0.05028 Brier 0.07457
platt a=1.5055970696466254 b=-1.6727252503737349
```

## Final test evaluation (compare_models.py) - identical to final_test_metrics.json
```
Full Ensemble (XGBoost): thr 0.5000 acc 0.9211 prec 0.9357 rec 0.9043 F1 0.9197 ROC-AUC 0.9708 PR-AUC 0.9759
Graph-Only: 0.5800 / 0.9035 ROC 0.9488 | Sequence-Only: 0.4900 / 0.8711 ROC 0.9438 | Random Forest: 0.4700 / 0.8234 ROC 0.9065
SHAP matrix dimension: (20172, 7) (7 features).
```

## Known failure at the end of the run
`scripts/audit_measurements.py` (full mode) crashed with `FileNotFoundError: models/experiments/features_cache.npz`, so
`audit_after_full.json` (fresh SHAP / ECE / Brier) was never produced.
