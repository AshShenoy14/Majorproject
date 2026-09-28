# Step 5 — Final Held-Out Test Set Evaluation and Scientific Comparison Report

**Artifact:** `ESMGAT/results/final_evaluation_report.md`  
**Execution Context:** Git branch `gat-exp`, Step 5 (Final Scientific Evaluation)  
**Evaluation Nature:** Strictly Read-Only / No Model Retraining / No Threshold Tuning on Test  

---

## 1. Evaluation Dataset & Sample Count

* **Held-Out Test Dataset:** `data/processed/test.csv`
* **Test Sample Count:** Exactly **20,172** protein pairs.
* **Positive Pairs:** 10,086
* **Negative Pairs:** 10,086 (balanced 1:1 ratio)
* **Unique Proteins in Test Set:** 10,278 proteins.
* **Feature Coverage:** 100% of test proteins are mapped in `data/processed/embeddings.pt` (ESM-2 representation) and `data/processed/ppi_graph_mapping.pt` (STRING network topology).

---

## 2. Leakage Audit & Test Set Isolation

* **Training Set:** `data/processed/train.csv` (161,369 pairs).
* **Validation Set:** `data/processed/val.csv` (20,171 pairs).
* **Overlap Check:** Pairwise symmetric overlap between `train.csv` and `test.csv` was verified:
  $$\text{Overlap}(\text{Train}, \text{Test}) = 0 \text{ pairs (0.000\%)}$$
* **Isolation Verification:** `test.csv` was **not** used for:
  * Model architecture design or weight updates
  * Early stopping or model checkpoint selection
  * Platt calibration parameter fitting ($a, b$)
  * Classification threshold selection
  * XGBoost hyperparameter tuning or stacking meta-learner training
  * SHAP feature selection (SHAP was conducted strictly as a post-hoc analysis on frozen model outputs)

---

## 3. Threshold Methodology

Classification decision thresholds were established **strictly prior to test inference**, maintaining rigorous methodological hygiene:

| Model | Evaluation Threshold | Threshold Source / Provenance |
| :--- | :---: | :--- |
| **ESM-MLP** | **0.45** | Validation $F_1$-max sweep on `val.csv` (`esm150m_results/assets/evaluation/final_test_metrics.json`) |
| **GraphSAGE** | **0.61** | Validation $F_1$-max sweep on `val.csv` (`esm150m_results/assets/evaluation/final_test_metrics.json`) |
| **GAT** | **0.50** | Standard validation & OOF calibration evaluation threshold (`ESMGAT/training/train_gat_colab.py`, line 345; `ESMGAT/results/gat_val_metrics.json`) |
| **ESM + GraphSAGE + XGBoost** | **0.45** | Validation $F_1$-max sweep on `val.csv` (`esm150m_results/assets/evaluation/final_test_metrics.json`) |
| **ESM + GAT + XGBoost** | **0.50** | OOF meta-learner training & evaluation threshold (`ESMGAT/training/train_gat_ensemble.py`; `ESMGAT/results/gat_oof_metrics.json`) |

*(Note: Supplementary metrics at standard threshold 0.50 for all systems are also recorded below for completeness).*

---

## 4. Calibration Methodology

Graph branch link prediction scores are calibrated using separate, model-specific univariate Platt scaling ($P(y=1|z) = \frac{1}{1 + \exp(-(a \cdot z + b))}$):

* **GraphSAGE Calibrator:** `models/graph_calibrator.json`
  * Fitted during GraphSAGE validation on `val.csv`.
  * Platt parameters: $a \approx 6.012$, $b \approx -0.731$.
  * SHA256: `dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd`.
* **GAT Calibrator:** `ESMGAT/weights/gat_calibrator.json`
  * Fitted during GAT validation on `val.csv`.
  * Platt parameters: $a = 65.1508$, $b = 6.6579$.
* Neither calibrator was re-fitted or modified using test data. The GraphSAGE calibrator was never used for GAT, and the GAT calibrator was never used for GraphSAGE.

---

## 5. Five-Model Comparison on Held-Out Test Set (20,172 Pairs)

### Table 1: Primary Metrics (Validation-Derived Thresholds)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Threshold | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ESM-MLP** | 88.42% | 0.8667 | 0.9082 | 0.8870 | 0.9530 | 0.9569 | 0.45 | 9,160 | 8,677 | 1,409 | 926 |
| **GraphSAGE** | 91.62% | 0.9355 | 0.8940 | 0.9143 | 0.9562 | 0.9681 | 0.61 | 9,017 | 9,464 | 622 | 1,069 |
| **GAT** | 76.30% | 0.7678 | 0.7541 | 0.7609 | 0.8487 | 0.8665 | 0.50 | 7,606 | 7,786 | 2,300 | 2,480 |
| **ESM + GraphSAGE + XGBoost** | 93.10% | 0.9448 | 0.9155 | 0.9300 | 0.9753 | 0.9801 | 0.45 | 9,234 | 9,547 | 539 | 852 |
| **ESM + GAT + XGBoost** | 89.79% | 0.9205 | 0.8710 | 0.8951 | 0.9582 | 0.9627 | 0.50 | 8,785 | 9,327 | 759 | 1,301 |

### Table 2: Supplementary Metrics (Uniform Threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ESM-MLP** | 88.72% | 0.9039 | 0.8665 | 0.8848 | 0.9530 | 0.9569 | 0.50 |
| **GraphSAGE** | 91.14% | 0.9103 | 0.9128 | 0.9115 | 0.9562 | 0.9681 | 0.50 |
| **GAT** | 76.30% | 0.7678 | 0.7541 | 0.7609 | 0.8487 | 0.8665 | 0.50 |
| **ESM + GraphSAGE + XGBoost** | 93.02% | 0.9517 | 0.9064 | 0.9285 | 0.9753 | 0.9801 | 0.50 |
| **ESM + GAT + XGBoost** | 89.79% | 0.9205 | 0.8710 | 0.8951 | 0.9582 | 0.9627 | 0.50 |

---

## 6. GraphSAGE vs. GAT Architecture & Implementation Comparison

Both graph models operate on the STRING protein-protein interaction graph and share identical input node feature dimensionality, decoder architecture, and evaluation protocol:

| Architectural Component | GraphSAGE (Production) | GAT (Experimental) | Implementation Parity |
| :--- | :--- | :--- | :--- |
| **Input Feature Dimension** | **643** (640 ESM-2 + 3 network topology) | **643** (640 ESM-2 + 3 network topology) | **Identical** (both verified from checkpoints) |
| **Graph Node Count** | 12,323 nodes | 12,323 nodes | **Identical** |
| **Graph Edge Count** | 161,370 directed edges | 161,370 directed edges | **Identical** |
| **Aggregation Mechanism** | Inductive mean neighborhood aggregation (`SAGEConv`) | Multi-head self-attention coefficients (`GATConv`, 4 heads, $d_k=64$) | **Divergent** (GraphSAGE uniform mean vs GAT dynamic attention) |
| **Hidden Layer Dimension** | 256 | 256 | **Identical** |
| **Number of Layers** | 2 layers (`conv1`, `conv2`) | 2 layers (`conv1`, `conv2`) | **Identical** |
| **Normalization / Dropout** | `LayerNorm` (input) + `GNNBatchNorm` + ReLU + Dropout | `LayerNorm` (input) + `GNNBatchNorm` + ReLU + Dropout | **Identical** |
| **Link Decoder Head** | Bilinear + Concat MLP: $[u, v, \|u-v\|, u \odot v, \text{Bilinear}(u,v)]$ | Bilinear + Concat MLP: $[u, v, \|u-v\|, u \odot v, \text{Bilinear}(u,v)]$ | **Identical** (1,025-d input $\to$ 512 $\to$ 256 $\to$ 1) |
| **Calibrator** | PlattScaler ($a \approx 6.01, b \approx -0.73$) | PlattScaler ($a = 65.15, b = 6.66$) | **Independent** |
| **Stacking Meta-Learner** | XGBoost (500 trees, depth 7, $\eta=0.01$) | XGBoost (500 trees, depth 7, $\eta=0.01$) | **Identical Hyperparameters** |
| **Meta-Features** | 7 meta-features (`[p_seq, p_sage, ...]`) | 7 meta-features (`[p_seq, p_gat, ...]`) | **Identical Mathematical Formulation** |

---

## 7. GAT Validation vs. OOF Discrepancy Investigation

A documented discrepancy was analyzed between earlier Step 3 validation (ROC-AUC 0.8554, PR-AUC 0.8735) and Step 4 5-fold OOF (raw ROC-AUC 0.5941, calibrated ROC-AUC 0.7828):

1. **Test Set Verification:** On the held-out test set (`test.csv`), the frozen base GAT model achieved **ROC-AUC = 0.8487** and **PR-AUC = 0.8665**, closely mirroring the 0.8554 validation result under identical full-graph conditions.
2. **Global Raw Logit Distortion (0.5941):** In Step 4, 5 separate models trained independently had differing baseline logit offsets. While individual folds achieved raw AUCs of 0.7727, 0.8445, 0.5683, 0.8356, and 0.8064, directly pooling uncalibrated logits caused cross-fold rank distortion. Fold-specific Platt scaling resolved this, bringing calibrated OOF AUC to **0.7828**.
3. **Graph Sub-Sampling Impact (0.8554 vs 0.7828):** In 5-fold OOF, all positive edges belonging to held-out validation pairs were strictly excised from the message-passing graph to prevent transductive leakage. Each fold model also trained on 28% less data and terminated after only 2–4 epochs (compared to 18 epochs for the base model).
4. **Conclusion:** The difference reflects the stricter topological masking and reduced epoch budget of the out-of-fold procedure rather than an implementation fault or test leakage.

Full investigation details are saved in: `ESMGAT/results/gat_validation_vs_oof_analysis.md`.

---

## 8. GAT Ensemble SHAP Analysis

SHAP analysis was performed using `shap.TreeExplainer` on the frozen GAT XGBoost meta-learner (`ESMGAT/weights/gat_ensemble_model.pkl`) across all 20,172 held-out test pairs:

| Rank | Meta-Feature | Description | Mean Absolute SHAP Value |
| :---: | :--- | :--- | :---: |
| **1** | `p_seq` | Sequence model probability | **1.463925** |
| **2** | `consensus` | Joint product $p_{\text{seq}} \times p_{\text{gat}}$ | **1.167934** |
| **3** | `conf_seq` | Sequence confidence $\|p_{\text{seq}} - 0.5\|$ | **0.164674** |
| **4** | `p_gat` | Calibrated GAT probability | **0.150359** |
| **5** | `max_conf` | Dominant branch confidence $\max(\text{conf}_{\text{seq}}, \text{conf}_{\text{gat}})$ | **0.083783** |
| **6** | `diff` | Branch disagreement $\|p_{\text{seq}} - p_{\text{gat}}\|$ | **0.045858** |
| **7** | `conf_gat` | GAT confidence $\|p_{\text{gat}} - 0.5\|$ | **0.023927** |

### Evidence-Based Interpretation
* **Attribution Finding:** SHAP values indicate that `p_seq` (mean |SHAP| = 1.4639) and the multiplicative `consensus` term (mean |SHAP| = 1.1679) contributed most strongly to the predictions of the trained XGBoost meta-learner.
* **Secondary Contribution:** Direct GAT probability `p_gat` contributed moderately (mean |SHAP| = 0.1504), with non-linear confidence features providing minor residual refinement.
* **Scientific Caveat:** These values reflect feature attribution within the trained tree structure of the meta-learner and must not be interpreted as biological causation.

---

## 9. Artifact Integrity Verification (SHA256 Hashes)

All frozen production artifacts were verified before and after evaluation:

| Artifact Path | Expected SHA256 Hash | Actual Verified SHA256 Hash | Status |
| :--- | :--- | :--- | :---: |
| `models/graph_model_best.pth` | `16d3b16ea464f5e1bd435c8c2bd3aac2cf2415a8f50fb03745f87bd4aa5d57fa` | `16d3b16ea464f5e1bd435c8c2bd3aac2cf2415a8f50fb03745f87bd4aa5d57fa` | **MATCH** |
| `models/ensemble_model.pkl` | `7e2c9543cb16f19429292b449128e03ace1bd88cc115c1db79fa29f1c8e98b60` | `7e2c9543cb16f19429292b449128e03ace1bd88cc115c1db79fa29f1c8e98b60` | **MATCH** |
| `models/graph_calibrator.json` | `dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd` | `dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd` | **MATCH** |

---

## 10. Summary of Generated Step 5 Artifacts

All Step 5 outputs are located strictly under `ESMGAT/`:

1. `ESMGAT/training/evaluate_gat.py` — Evaluation and SHAP execution pipeline.
2. `ESMGAT/results/final_predictions.csv` — Predictions for all 20,172 test pairs.
3. `ESMGAT/results/final_comparison.csv` — Comparative metrics table.
4. `ESMGAT/results/final_test_metrics.json` — Detailed JSON metrics with TP/TN/FP/FN and metadata.
5. `ESMGAT/results/gat_validation_vs_oof_analysis.md` — Root-cause report on validation vs OOF numbers.
6. `ESMGAT/results/gat_shap_importance.csv` — Global SHAP feature importance table.
7. `ESMGAT/results/gat_shap_summary.png` — High-resolution SHAP summary plot.
8. `ESMGAT/results/shap/gat_shap_summary.png` — SHAP summary plot in dedicated subdirectory.
9. `ESMGAT/results/shap/gat_shap_values.npz` — Saved raw SHAP values matrix and base value.
10. `ESMGAT/results/final_evaluation_report.md` — This comprehensive scientific evaluation report.

---

## 11. Limitations

1. **Computational Platform:** Test evaluation and SHAP analysis were executed on CPU due to local hardware availability; all outputs are deterministic given frozen model weights.
2. **Attention Head Capacity:** GAT utilized 4 attention heads with 64 dimensions each (256 hidden channels total) to maintain strict layer-width parity with GraphSAGE; higher head counts or deeper attention layers were outside the frozen Step 4 scope.
3. **Transductive PPI Setting:** Both graph models operate on fixed STRING network nodes where protein identities are known during inference, reflecting standard transductive PPI benchmarking rather than unseen-protein link prediction.
