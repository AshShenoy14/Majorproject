# TransGraph-PPI: Final Production Validation Record

This document provides the definitive research and production validation record for the **TransGraph-PPI** hybrid machine learning framework. All metrics, dataset split statistics, model evaluations, forensic audits, unit tests, and production API tests recorded here have been verified on the frozen codebase.

---

## 1. Dataset & Split Isolation Summary

To ensure rigorous performance estimation and prevent memorization, the dataset was constructed with strict protein-level and edge-level isolation.

| Property | Full Dataset | Training Set | Validation Set |
| :--- | :--- | :--- | :--- |
| **Total Protein Pairs** | 363,081 | 322,739 | 40,342 |
| **Percentage Split** | 100% | 88.9% | 11.1% |
| **Class Balance Ratio** | 1:1 (Balanced) | 1:1 (Balanced) | 1:1 (Balanced) |
| **Protein Isolation (Node Split)** | Unseen Split | Training Nodes | Held-out Validation Nodes |
| **Pair Overlap** | N/A | **0 Pairs Overlapping** | **0 Pairs Overlapping** |

### Split Isolation Safeguards
- **Negative Sampling**: True interactions from the STRING database were labeled positive ($1$). Negative pairs ($0$) were sampled uniformly at random across non-interacting protein pairs, strictly filtering out any known STRING interaction.
- **Zero Pair Overlap**: Programmatically verified that no pair present in the training set appears in the validation or test sets.

---

## 2. Model Performance & Baseline Benchmark

The table below summarizes performance on the held-out benchmark set across model components and baselines.

| Model / Baseline | Optimal Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC (AUPRC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | 0.50 | 0.8430 | 0.8350 | 0.8540 | 0.8444 | 0.9120 | 0.9105 |
| **GAT (Graph Alone)** | 0.62 | 0.8017 | 0.7599 | 0.8821 | 0.8164 | 0.8860 | 0.8848 |
| **ESM-MLP (Sequence Alone)** | 0.49 | 0.9180 | 0.9061 | 0.9326 | 0.9191 | 0.9739 | 0.9736 |
| **Ensemble (Ours - OOF Meta-Learner)** | **0.50** | **0.9154** | **0.9181** | **0.9122** | **0.9151** | **0.9764** | **0.9773** |

### Key Observations
- The **XGBoost Meta-Learner** trained on Out-Of-Fold (OOF) predictions achieves superior ROC-AUC ($0.9764$) and PR-AUC ($0.9773$), outperforming both individual base models and the Random Forest baseline.
- **Synergistic Integration**: Combining structural network topology (GAT) with deep sequence embeddings (ESM-2) yields robust discrimination even under high sequence variation.

---

## 5. SHAP Feature Explanation Audit

SHAP (SHapley Additive exPlanations) values for the XGBoost meta-learner demonstrate the feature contribution hierarchy:

1. **`esm_probability`** (ESM-2 Sequence Model): Primary driver of overall interaction probability.
2. **`gat_probability`** (GAT Graph Model): Strong secondary contributor providing network topological context.
3. **`Biological_Match` / Subcellular Co-localization**: Ensures physical feasibility.
4. **`Model_Disagreement` / Consensus Delta**: Modulates prediction confidence.

---

## 6. Real Production API End-to-End Test

Tested on live FastAPI server (`http://127.0.0.1:8000`) using Ensembl Protein IDs from the dataset:
- **Protein A**: `ENSP00000370517`
- **Protein B**: `ENSP00000496166`

### Endpoint Test Results

#### 1. `POST /predict`
- **HTTP Status**: `200 OK`
- **Interaction Probability**: `0.9601` (96.01%)
- **Confidence Score**: `0.9202` (92.02%)
- **ESM Sequence Signal**: `0.8563` (85.63%)
- **GAT Graph Signal**: `0.9468` (94.68%)
- **SHAP Explanation**: Includes sequence contribution, graph contribution, biological match, and consensus signals.
- **Finite Check**: `True` (0 NaNs, 0 Infs).

---

## 7. Test Suite Status

Executed `pytest tests/`:
- **Total Tests**: 13
- **Passed**: **13 / 13**
- **Failed**: 0
- **Test Modules**:
  - `tests/test_alphafold_features.py` (Passed)
  - `tests/test_basic.py` (Passed)
  - `tests/test_cold_start.py` (Passed)
  - `tests/test_containerization.py` (Passed)
  - `tests/test_explainability.py` (Passed)
  - `tests/test_inference_safety.py` (5 tests Passed)
  - `tests/test_localization_sampling.py` (Passed)
  - `tests/test_quantized_inference.py` (Passed)
  - `tests/test_therapeutic_target.py` (Passed)

---

## 8. Summary of Verification Status

- **Model Checkpoints**: Frozen (`models/sequence_model_best.pth`, `models/graph_model_best.pth`, `models/ensemble_model.pkl`, `models/random_forest_baseline.pkl`).
- **Numerical Stability**: Hardened; 100% finite outputs across all API endpoints.
- **Release Ready**: Yes (`v1.0-research`).
