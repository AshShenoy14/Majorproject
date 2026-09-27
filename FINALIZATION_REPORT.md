# TransGraph-PPI: Final Project Cleanup and Verification Report

**Project**: TransGraph-PPI (Hybrid Deep Learning Framework for Protein–Protein Interaction Prediction)  
**Branch**: `gat-exp`  
**Date**: September 27, 2026  
**Status**: COMPLETE, VERIFIED & FROZEN

---

## 1. Executive Summary & GATv2 Deletion Confirmation

In accordance with project specifications:
- **GATv2 has been completely and irrevocably purged** from the project repository.
- The Step 6 GATv2 improvement experiment was an exploratory investigation that failed to improve controlled validation / OOF metrics over the primary architecture.
- **Zero active source code, test files, configuration files, reports, documentation, or frontend components contain references to GATv2.**
- The project retains **strictly two graph architectures**:
  1. **GraphSAGE** — **Primary Final Model** (`ESM-2 + GraphSAGE + XGBoost + SHAP`)
  2. **Standard GAT** — **Controlled Experimental Comparison** (`ESM-2 + Standard GAT + XGBoost + SHAP`)

---

## 2. Inventory of Deleted Files

The entire directory `ESMGAT/step6_gat_improvement/` and all its subdirectories were permanently deleted:
- `ESMGAT/step6_gat_improvement/models/` (including `gatv2_model.py`, `gatv2_ensemble.py`)
- `ESMGAT/step6_gat_improvement/training/` (including `evaluate_gatv2.py`, `train_gatv2_ensemble.py`)
- `ESMGAT/step6_gat_improvement/checkpoints/` (including OOF checkpoints `gatv2_fold*.npz`)
- `ESMGAT/step6_gat_improvement/weights/` (including `gatv2_model_best.pth`, `gatv2_calibrator.json`, `gatv2_ensemble_model.pkl`)
- `ESMGAT/step6_gat_improvement/results/` (including reports, comparison CSVs, plots, SHAP arrays)
- `ESMGAT/step6_gat_improvement/logs/`
- `ESMGAT/step6_gat_improvement/tests/` (including `test_gatv2_pipeline.py`)

Global repository search (`grep_search`) confirmed **0 matches** for `gatv2`, `GATv2`, `GATv2Conv`, `step6_gat_improvement`, and `S6-GATV2-01`.

---

## 3. Preserved Artifacts Inventory

### Core Production GraphSAGE Artifacts (Untouched & Frozen)
- `models/sequence_model_best.pth`
- `models/graph_model_best.pth`
- `models/ensemble_model.pkl`
- `models/graph_calibrator.json`
- `checkpoints/graph_checkpoint.pt`
- `checkpoints/oof/fold*.npz`
- `src/models/graph_model.py`
- `src/training/train_graph_model.py`
- `src/training/base_trainers.py`
- `src/training/train_ensemble.py`

### Standard GAT Controlled Comparison Artifacts (Validated & Preserved)
- `ESMGAT/models/gat_model.py`
- `ESMGAT/models/gat_ensemble.py`
- `ESMGAT/training/gat_base_trainers.py`
- `ESMGAT/training/train_gat_colab.py`
- `ESMGAT/training/train_gat_ensemble.py`
- `ESMGAT/weights/gat_model_best.pth`
- `ESMGAT/weights/gat_calibrator.json`
- `ESMGAT/weights/gat_ensemble_model.pkl`
- `ESMGAT/checkpoints/oof/gat_fold1.npz` through `gat_fold5.npz`
- `ESMGAT/results/final_predictions.csv`
- `ESMGAT/results/final_test_metrics.json`
- `ESMGAT/results/gat_shap_importance.csv`
- `ESMGAT/results/gat_shap_summary.png`
- `ESMGAT/backend/main.py` (Isolated GAT microservice)

---

## 4. Production Artifact Cryptographic Verification (SHA256)

All production models and calibrators were cryptographically verified against their canonical frozen hashes:

| File Path | Expected SHA256 Hash | Actual Verified SHA256 Hash | Status |
| :--- | :--- | :--- | :---: |
| `models/graph_model_best.pth` | `16d3b16ea464f5e1bd435c8c2bd3aac2cf2415a8f50fb03745f87bd4aa5d57fa` | `16d3b16ea464f5e1bd435c8c2bd3aac2cf2415a8f50fb03745f87bd4aa5d57fa` | **MATCH (VERIFIED)** |
| `models/ensemble_model.pkl` | `7e2c9543cb16f19429292b449128e03ace1bd88cc115c1db79fa29f1c8e98b60` | `7e2c9543cb16f19429292b449128e03ace1bd88cc115c1db79fa29f1c8e98b60` | **MATCH (VERIFIED)** |
| `models/graph_calibrator.json` | `dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd` | `dea5cb2a650f5ea85686aa905a5d99c26511b8c081bcf2efea193a76a9657bcd` | **MATCH (VERIFIED)** |

---

## 5. Standard GAT Artifact Verification

| Artifact | File Size | Checksum Prefix | Status |
| :--- | :---: | :---: | :---: |
| `ESMGAT/weights/gat_model_best.pth` | 4,641,041 bytes | `46213dfde2b8ba0c...` | Valid |
| `ESMGAT/weights/gat_calibrator.json` | 75 bytes | `18adc2361987c6bb...` | Valid |
| `ESMGAT/weights/gat_ensemble_model.pkl` | 3,173,919 bytes | `2890b91c2417dd6e...` | Valid |
| `ESMGAT/checkpoints/oof/gat_fold1.npz` | 1,035,492 bytes | `19f2015c324f18ff...` | Valid |
| `ESMGAT/checkpoints/oof/gat_fold2.npz` | 1,035,492 bytes | `a93d1699b24712d6...` | Valid |
| `ESMGAT/checkpoints/oof/gat_fold3.npz` | 1,035,492 bytes | `67a5a7cf6db01dd7...` | Valid |
| `ESMGAT/checkpoints/oof/gat_fold4.npz` | 1,035,492 bytes | `fd0365294a64e7c5...` | Valid |
| `ESMGAT/checkpoints/oof/gat_fold5.npz` | 1,035,460 bytes | `6b19daca7818a10e...` | Valid |

---

## 6. Research Paper Updates (`paper.tex`)

The manuscript was updated with full academic rigor and project consistency:
1. **Section IV — Graph Architecture Selection Subsection**:
   - Added `\subsection{Graph Architecture Selection}` documenting that GAT was initially proposed and subsequently evaluated side-by-side with GraphSAGE under identical ESM-2 embeddings, graph feature extraction, and XGBoost stacking. GraphSAGE was selected as the primary architecture based on empirical efficiency and interaction prediction performance, while Standard GAT was maintained as a controlled experimental baseline.
   - Neutral scientific language preserved: avoids claiming "GAT failed" or making generalized claims beyond the scope of the project.
2. **Section V — Table I (`tab:performance`)**:
   - Includes `ESM-2 + GAT + XGBoost` (*Controlled Experimental Comparison*): Accuracy 89.79%, Precision 0.9205, Recall 0.8710, F1 0.8951, ROC-AUC 0.9582, PR-AUC 0.9627.
   - Includes `ESM-2 + GraphSAGE + XGBoost` (*Primary Final Model*): Accuracy 93.10%, Precision 0.9448, Recall 0.9155, F1 0.9300, ROC-AUC 0.9753, PR-AUC 0.9801.
   - Includes standalone GAT base model (76.30% Acc, 0.7609 F1, 0.8487 ROC-AUC).
3. **Explainability & SHAP Framing**:
   - Explicitly clarified that SHAP explains feature attribution across the 7 meta-features of the XGBoost stacking ensemble and does not claim biological mechanistic causation.
4. **Cleanliness**:
   - Zero occurrences of GATv2 in `paper.tex`.

---

## 7. Model Comparison & Final Evaluation Summary

Evaluated on the exact 20,172 held-out pairs from `data/processed/test.csv`:

| Model Architecture | Role | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ESM-2 + GraphSAGE + XGBoost** | **Primary Final Model** | **93.10%** | **0.9448** | **0.9155** | **0.9300** | **0.9753** | **0.9801** |
| **ESM-2 + GAT + XGBoost** | **Controlled Comparison** | **89.79%** | **0.9205** | **0.8710** | **0.8951** | **0.9582** | **0.9627** |
| ESM-MLP (Sequence Only) | Base Sequence Model | 88.42% | 0.8667 | 0.9082 | 0.8870 | 0.9530 | 0.9569 |
| GraphSAGE Base (Calibrated) | Base Graph Model | 91.62% | 0.9355 | 0.8940 | 0.9143 | 0.9562 | 0.9681 |
| GAT Base (Calibrated) | Base Graph Comparison | 76.30% | 0.7681 | 0.7538 | 0.7609 | 0.8487 | 0.8665 |
| Random Forest Baseline | Classical Baseline | 83.74% | 0.8380 | 0.8365 | 0.8373 | 0.9163 | 0.9222 |

---

## 8. Backend Architecture & Isolation

### GraphSAGE Backend (Port 8000 — Primary Production Service)
- **Path**: `app/backend/main.py`
- **Port**: `8000`
- **Endpoints**: `/predict`, `/predict_batch`, `/evaluation/final`, `/network/subgraph`, `/targets/therapeutic`, etc.
- **Independence**: Fully self-contained. Completely unaware of and unaffected by the state of port 8001.

### Standard GAT Backend (Port 8001 — Isolated Comparison Service)
- **Path**: `ESMGAT/backend/main.py`
- **Port**: `8001`
- **Endpoints**:
  - `GET /health` — Reports service health, loaded GAT models, and calibrator parameters.
  - `GET /info` — Metadata confirming architecture, parameters, and comparison role.
  - `POST /predict` — Standard GAT ensemble prediction with SHAP meta-feature attributions.
  - `POST /predict_batch` — High-throughput batch prediction.
- **Fail-Safe Operation**: If port 8001 is offline, the frontend notifies the user while GraphSAGE predictions continue without disruption.

---

## 9. Frontend Enhancements (`app/frontend/`)

### 1. Dual-Architecture Model Selector
- Dropdown selector on the Predict view:
  - `GraphSAGE Ensemble (Primary Final Model)` — **Default**
  - `GAT Ensemble (Controlled Comparison)`
- Displays descriptive role badges:
  - GraphSAGE: `"Primary Final Model"`
  - GAT: `"Controlled Comparison — Original Proposal Architecture"`

### 2. Prediction Flow & Thresholds
- Uses authentic validation-derived thresholds:
  - GraphSAGE Ensemble threshold: `0.45`
  - GAT Ensemble threshold: `0.50`
- Clear status classification: `"Likely Interaction"` vs `"Unlikely Interaction"`.
- Displays calibrated probability and confidence gauge.

### 3. SHAP Interpretability
- Shows real-time attribution breakdown across the 7 meta-features (`p_seq`, `p_graph`, `conf_seq`, `conf_graph`, `diff`, `max_conf`, `consensus`).
- Includes clear scientific disclaimer that SHAP measures model feature contributions and does not prove biological causation.

### 4. Side-by-Side Model Comparison Section
- Renders the controlled performance comparison table directly within the UI (Accuracy, F1, ROC-AUC, PR-AUC).

### 5. Protein 3D Structure Viewer (`Protein3DView.jsx`)
- Powered by `pdbe-molstar`.
- Supports rotation, scroll zoom, and a dedicated **"Reset View"** button.
- Highlights binding interface regions in emerald green and focused residues in rose red.
- **Clean Structure Unavailable Fallback**: When experimental coordinates are unavailable, renders an informative badge and note without error or fabrication.

---

## 10. Automated Test Results

All test suites were executed using the dedicated project virtual environment (`.venv\Scripts\python.exe`):

1. **`tests/test_inference_safety.py`**:
   - `7 passed in 10.64s`
   - Verified 7-feature parity, rejection of stale 8-feature models, Platt calibration integrity, and checkpoint existence.
2. **`tests/test_gat_pipeline.py`**:
   - `16 passed in 82.53s`
   - Verified GAT dimension parameters, 4-head attention, decoder shape (1025), synthetic graph forward pass, frozen GraphSAGE hash checks, path safety assertions, 5-fold OOF coverage (161,369 unique samples), and SHAP array integrity.
3. **`tests/test_gat_backend.py`**:
   - `3 passed in 32.89s`
   - Verified `/health`, `/info`, and `/predict` endpoints on the isolated GAT backend.
4. **`tests/test_all_endpoints.py`**:
   - `13 passed in 41.24s`
   - Verified all production GraphSAGE API routes, batch prediction, centrality, network subgraph, and chat.
5. **Frontend Production Build (`npm run build`)**:
   - `17504 modules transformed`, `built in 2m 34s`, exit code `0`.

---

## 11. Final Demonstration & Execution Instructions

### Step 1: Start the Primary GraphSAGE Backend (Port 8000)
```bash
python -m uvicorn app.backend.main:app --port 8000 --reload
```

### Step 2: (Optional) Start the Standard GAT Comparison Backend (Port 8001)
```bash
python ESMGAT/backend/main.py --port 8001
```

### Step 3: Start the React Frontend Dashboard
```bash
cd app/frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

### Demo Highlights:
1. Select **GraphSAGE Ensemble (Primary Final Model)** and click **"Predict Interaction"** for any pair (e.g., TP53 `ENSP00000269305` vs MDM2 `ENSP00000258149`).
2. Switch to **GAT Ensemble (Controlled Comparison)** to view the comparative prediction and 7-feature SHAP attribution.
3. Inspect the **Model Comparison Table** displaying the final 93.10% vs 89.79% benchmark.
4. Explore the **3D Protein Structure Viewer** with rotation, zoom, and interaction interface highlighting.
