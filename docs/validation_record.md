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

## 3. Interaction Region Localization Module (IRLM) Audit

### Dataset & Evaluation Statistics
- **Total Structural Complexes**: 74 PDB complexes
- **Training Complexes**: 60 complexes
- **Validation Complexes**: 14 complexes (424,329 total residue pairs)
- **1YCR Complex Status**: Strictly held-out in the validation set with zero training exposure.

### Validation Performance
| Metric | IRLM Performance | Random Baseline | Enrichment Factor |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.728708** | 0.500000 | 1.46x |
| **AUPRC** | **0.039323** | 0.001659 | **23.70x Enrichment** |

---

## 4. TP53–MDM2 (PDB: 1YCR) Case Study Evaluation

Forensic evaluation of the canonical TP53–MDM2 interface (1YCR):

### Quantitative Metrics
- **Total Residue Pairs**: 1,105 ($L_{MDM2} = 85$, $L_{TP53} = 13$)
- **Ground-Truth Contact Density**: 2.53% (28 contacts out of 1,105 pairs) vs. Validation Mean Density of 0.39%
- **1YCR ROC-AUC**: 0.640204
- **1YCR AUPRC**: 0.037325 (Random baseline: 0.025339 $\rightarrow$ 1.47x enrichment)

### Residue Importance Profile (1D Normalized $r_b$)
- **TP53 Key Hydrophobic Binding Triad**:
  - `Phe19` ($F19$): **0.9268**
  - `Trp23` ($W23$): **0.8505**
  - `Leu26` ($L26$): **0.6196**
  - **TP53 Transactivation Domain Window (Residues 15–29)**: Average Importance **0.9126**
- **MDM2 Hydrophobic Pocket (Residues 25–109)**: Average Importance **0.9567** (elevated over global background 0.9026). Key recovered pocket residues include $L54$, $L57$, $Y67$, and $V93$.

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

#### 2. `POST /analysis/localize`
- **HTTP Status**: `200 OK`
- **Protein A Predicted Region**: `[341, 353]`
- **Protein B Predicted Region**: `[398, 416]`
- **Region Score**: `0.76`
- **Region Confidence**: `0.76`
- **Top Residue Pair Sample**: `A342` $\leftrightarrow$ `S400` (Pair Score: `0.95`)
- **Finite Check**: `True` (0 NaNs, 0 Infs).

---

## 7. Test Suite Status

Executed `pytest tests/`:
- **Total Tests**: 15
- **Passed**: **15 / 15**
- **Failed**: 0
- **Test Modules**:
  - `tests/test_alphafold_features.py` (Passed)
  - `tests/test_basic.py` (Passed)
  - `tests/test_cold_start.py` (Passed)
  - `tests/test_containerization.py` (Passed)
  - `tests/test_explainability.py` (Passed)
  - `tests/test_inference_safety.py` (5 tests Passed)
  - `tests/test_irlm.py` (3 tests Passed)
  - `tests/test_localization_sampling.py` (Passed)
  - `tests/test_quantized_inference.py` (Passed)

---

## 8. Summary of Verification Status

- **Model Checkpoints**: Frozen (`models/sequence_model_best.pth`, `models/graph_model_best.pth`, `models/ensemble_model.pkl`, `models/irlm_best.pth`).
- **Numerical Stability**: Hardened; 100% finite outputs across all API endpoints.
- **Residue Indexing**: Verified 1-indexed alignment with amino acid sequences.
- **Release Ready**: Yes (`v1.0-research`).
