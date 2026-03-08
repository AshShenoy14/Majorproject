<p align="center">
  <img src="assets/banner_mp.png" alt="TransGraph-PPI Banner" width="100%">
</p>

# TransGraph-PPI  
### Hybrid Deep Learning Framework for Protein–Protein Interaction Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![React](https://img.shields.io/badge/Frontend-ReactJS-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A hybrid machine learning framework for **Protein–Protein Interaction (PPI) prediction** combining **Protein Language Models (ESM-2)** and **Graph Neural Networks (GAT)** with explainability and drug target prioritization.

---

# Overview

Protein–Protein Interactions play a crucial role in understanding biological systems, disease mechanisms, and drug discovery.

**TransGraph-PPI** integrates:

- Sequence Representation Learning
- Graph Structural Learning
- Ensemble Modeling
- Explainable AI

to build a robust prediction system.

The framework combines:

| Component | Role |
|-----------|------|
| ESM-2 | Extract protein sequence embeddings |
| MLP Sequence Model | Sequence-based interaction prediction |
| Graph Attention Network (GAT) | Learn interaction patterns in PPI networks |
| XGBoost Ensemble | Combine predictions from both models |
| SHAP | Explain model decisions |

---

# System Architecture

```mermaid
graph TD
    A[Protein Pair] --> B[ESM-2 Embedding]
    B --> C[Sequence Model - MLP]

    D[PPI Graph Network] --> E[GAT Model]

    C --> F[XGBoost Ensemble]
    E --> F

    F --> G[Interaction Probability]
    F --> H[SHAP Explainability]

    G --> I[Network Centrality Analysis]
    I --> J[Drug Target Prioritization]
```

---

# Model Performance

The framework was evaluated on the validation dataset. To ensure maximum reliability and balance, the prediction thresholds were tuned to maximize the F1-score across 5-fold cross-validation. 

**Table 1: Ablation Study and Baseline Comparison**  
This table serves as an ablation study — ESM-MLP and GAT in isolation quantify the individual contribution of sequence semantics and graph topology respectively, while the Ensemble measures synergistic gain. A standard Logistic Regression baseline is included to contextualize the value of the deep learning stack.

| Model | Optimal Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (baseline)| 0.5000 | 0.7120 | 0.7050 | 0.7200 | 0.7124 | 0.7450 | 0.7580 |
| Sequence Model (ESM-MLP) | 0.3700 | 0.8645 ± 0.0042| 0.8445 ± 0.0051| 0.8936 ± 0.0038| 0.8684 ± 0.0040| 0.9403 ± 0.0021| 0.9422 ± 0.0025|
| Graph Model (GAT) | 0.5500 | 0.7266 ± 0.0081| 0.7049 ± 0.0075| 0.7797 ± 0.0085| 0.7404 ± 0.0079| 0.8330 ± 0.0055| 0.8614 ± 0.0048|
| **Ensemble Model (Ours)** | **0.4400** | **0.8846 ± 0.0035**| **0.8831 ± 0.0041**| **0.8866 ± 0.0032**| **0.8848 ± 0.0031**| **0.9523* ± 0.0018**| **0.9566 ± 0.0015**|

*\*Note: The ROC-AUC of 0.9523 reported in the table reflects the average across all cross-validation folds on the validation set. Visualized curves (e.g., AUC = 0.968) in the generated plots may represent the absolute peak performance on a specific held-out test fold.*

The ensemble model substantially improves prediction performance and reliability by strategically integrating **sequence semantics** and **interaction network topology**.

---

# Dataset

The dataset contains **protein pairs and interaction labels** used to train the model.

| Property | Value |
|----------|------|
| Total Samples | 363,081 |
| Training Samples | 322,739 |
| Validation Samples | 40,342 |
| Class Balance | Balanced |

Data sources include:

- STRING Database
- UniProt
- ChEMBL

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/TransGraph-PPI.git
cd TransGraph-PPI
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Frontend Dependencies

```bash
cd app/frontend
npm install
cd ../..
```

---

# Data Pipeline

Run the pipeline in sequence.

## Data Collection
```bash
python src/data/collect_ppi.py
```
Fetches:
- Protein interaction data
- Protein sequences
- Drug target information

---

## Data Preprocessing
```bash
python src/data/preprocess_data.py
```
**Negative Sampling Strategy:**
To construct a robust and balanced dataset (1:1 positive-to-negative ratio), true interacting pairs from the STRING database are labeled as positives (1). For negative samples (0), we employ a **randomized non-interaction sampling** strategy. Proteins are paired uniformly at random, and any pair that appears in the established positive STRING interaction set is strictly filtered out.

To prevent **data leakage**, the node split methodology guarantees that test-set proteins and edges are completely unseen during the model's training phase, ensuring the models learn generalizable biological interaction rules rather than memorizing the network.

---

## Feature Extraction

```bash
python src/data/feature_extraction.py
```

Generates **ESM-2 embeddings**.

---

## Graph Construction

```bash
python src/data/graph_construction.py
```

Builds the **protein interaction graph**.

---

# Model Training

## Train Sequence Model

```bash
python src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt
```

---

## Train Graph Model

```bash
python src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt
```

---

## Train Ensemble Model

```bash
python src/training/train_ensemble.py
```

---

# Running the Web Application

## Start Backend

```bash
cd app/backend
uvicorn main:app --reload
```

Backend runs on:

```
http://localhost:8000
```

---

## Start Frontend

```bash
cd app/frontend
npm run dev
```

Frontend runs on:

```
http://localhost:5173
```

---

# Project Structure

```
TransGraph-PPI
│
├── src
│   ├── data
│   │   ├── collect_ppi.py
│   │   ├── preprocess_data.py
│   │   ├── feature_extraction.py
│   │   └── graph_construction.py
│   │
│   ├── models
│   │   ├── sequence_model.py
│   │   └── gat_model.py
│   │
│   ├── training
│   │   ├── train_sequence_model.py
│   │   ├── train_graph_model.py
│   │   └── train_ensemble.py
│   │
│   └── evaluation
│
├── app
│   ├── backend
│   └── frontend
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   └── trained model checkpoints
│
├── notebooks
│   └── research experiments
│
└── README.md
```

---

# Future Work

- GAT attention visualization
- Integration with STRING API
- Graph Transformer Networks
- Cloud deployment with Docker

---

# Citation

If you use this repository in research:

```
@project{transgraph_ppi,
  title={TransGraph-PPI: Hybrid Ensemble Framework for Protein Interaction Prediction},
  author={Ashwini Shenoy B,Basil S},
  year={2026}
}
```

---

# Contributors

**Ashwini Shenoy B**  
**Basil S**


Major Project — Deep Learning for Bioinformatics

