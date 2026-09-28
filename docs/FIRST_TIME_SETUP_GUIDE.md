# 🚀 TransGraph-PPI: First-Time Setup & Training Guide

Complete step-by-step instructions for setting up and training the model from scratch on a new system.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or higher |
| **Node.js** | 18+ (for frontend) |
| **GPU (Recommended)** | NVIDIA GPU with CUDA support (embedding extraction is very slow on CPU) |
| **RAM** | Minimum 16 GB (32 GB recommended) |
| **Disk Space** | ~5 GB free for the project data, models and model cache (the ESM-2 weights are ~570 MB and `models/random_forest_baseline.pkl` ~200 MB), plus the Python and Node dependencies |
| **Internet** | Required for downloading STRING DB data (~80 MB + ~7 MB) and ESM-2 model weights |

---

## Step 1: Clone & Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd majorproject

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> **PyTorch with CUDA**: If you have an NVIDIA GPU, install PyTorch with CUDA support **before** running `pip install -r requirements.txt`:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Check your CUDA version with `nvidia-smi` and pick the right URL from [pytorch.org](https://pytorch.org/get-started/locally/).

### Install Frontend Dependencies
```bash
cd app/frontend
npm install
cd ../..
```

---

## Step 2: Data Pipeline (Run in Order)

The `data/` folder contents are **not on GitHub** (gitignored due to size). You must generate them.

### 2a. Download Raw Data from STRING DB

```bash
python src/data/collect_ppi.py
```

**What this does:**
- Downloads `9606.protein.links.v12.0.txt.gz` (~80 MB) → protein interaction links
- Downloads `9606.protein.sequences.v12.0.fa.gz` (~7 MB) → protein sequences
- Files are saved to `data/raw/`

> [!TIP]
> If the download is slow or fails, you can manually download from:
> - Links: https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
> - Sequences: https://stringdb-static.org/download/protein.sequences.v12.0/9606.protein.sequences.v12.0.fa.gz
>
> Place both files in the `data/raw/` folder.

### 2b. Preprocess Data (Generate Train/Val/Test Splits)

```bash
python src/data/preprocess_data.py
```

**What this does:**
- Reads the STRING interactions file and filters by confidence score (≥ 900)
- Generates negative samples (non-interacting pairs)
- Splits data into train/val/test (80/10/10)
- Saves `train.csv`, `val.csv`, `test.csv` to `data/processed/`

> [!NOTE]
> This step requires the **uncompressed** STRING links file. The script reads from `data/raw/9606.protein.links.v12.0.txt.gz`. If the uncompressed file `data/9606.protein.links.v12.0.txt` exists at the project root, the script may use that instead. Check `src/utils/paths.py` if you encounter path issues.

### 2c. Extract ESM-2 Embeddings (⏱️ Longest Step)

```bash
python src/data/feature_extraction.py
```

**What this does:**
- Downloads the ESM-2 protein language model (`facebook/esm2_t30_150M_UR50D`) from HuggingFace (~570 MB, automatic, cached in `.cache/`)
- Generates one 640-dimensional (mean-pooled, float16) embedding for every unique protein
- Saves `embeddings.pt` to `data/processed/`

> [!CAUTION]
> **This is the most time-consuming step!**
> - **With GPU**: ~2-4 hours depending on GPU
> - **Without GPU (CPU only)**: Can take 12-24+ hours
> - The output file (`embeddings.pt`) is small (~19 MB for 12,323 proteins); the time goes into running ESM-2
> - Make sure you have enough disk space before starting

> [!TIP]
> If you already have the `embeddings.pt` file from a teammate (e.g., via USB drive, Google Drive, etc.), you can **skip this step entirely** by placing the file directly in `data/processed/`.

### 2d. Construct PPI Graph

```bash
python src/data/graph_construction.py
```

**What this does:**
- Loads the embeddings and training interactions
- Builds a PyTorch Geometric graph with protein node features
- Saves `ppi_graph.pt` (~26 MB) and `ppi_graph_mapping.pt` to `data/processed/`

---

## Step 3: Train the Models (Run in Order)

### 3a. Train Sequence Model (ESM-MLP)

```bash
python src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt
```

**Optional flags:**
- `--epochs 10` (default: 10)
- `--batch_size 32` (default: 32)
- `--lr 0.001` (default: 0.001)

**Output:** `models/sequence_model_best.pth`

### 3b. Train Graph Model (GraphSAGE)

```bash
python src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt
```

**Optional flags:**
- `--epochs 100` (default: 100)
- `--lr 0.005` (default: 0.005)

**Output:** `models/graph_model_best.pth`

### 3c. Train Ensemble Model (XGBoost Stacking)

```bash
python src/training/train_ensemble.py
```

**What this does:**
- Loads both trained base models (sequence + graph)
- Generates predictions on the validation set
- Trains an XGBoost meta-learner to combine both models

**Output:** `models/ensemble_model.pkl`

---

## Step 4: Run the Web Application

### Start Backend (FastAPI)
```bash
cd app/backend
uvicorn main:app --reload
```
Backend will be available at `http://localhost:8000`

### Start Frontend (React + Vite)
```bash
cd app/frontend
npm run dev
```
Frontend will be available at `http://localhost:5173`

---

## ⚡ Quick Alternative: One-Command Pipeline

If you want to run everything automatically (data pipeline + training), use:

```bash
python scripts/run_pipeline.py
```

Or for a **quick test with limited data** (good for verifying setup works):

```bash
python scripts/run_pipeline.py --limit 500
```

> [!WARNING]
> The full pipeline (`run_pipeline.py` without `--limit`) will take several hours due to embedding extraction. The `--limit` flag runs with only 500 interactions for a quick sanity check.

---

## 📁 Expected Files After Complete Setup

After all steps, your project should have these key files:

```
data/
├── raw/
│   ├── 9606.protein.links.v12.0.txt.gz    (~80 MB)
│   └── 9606.protein.sequences.v12.0.fa.gz (~7 MB)
├── processed/
│   ├── train.csv                          (~5.3 MB)
│   ├── val.csv                            (~670 KB)
│   ├── test.csv                           (~670 KB)
│   ├── embeddings.pt                      (~19 MB)
│   ├── ppi_graph.pt                       (~33 MB)
│   ├── ppi_graph_mapping.pt               (~340 KB)
│   ├── sequences_cache.json               (~7.5 MB)
│   └── predicted_network.csv              (~5 MB, from scripts/build_predicted_network.py)
models/
├── sequence_model_best.pth                (~25 MB)
├── graph_model_best.pth                   (~4.6 MB)
├── graph_calibrator.json                  (<1 KB)
├── ensemble_model.pkl                     (~3 MB)
└── random_forest_baseline.pkl             (~200 MB, baseline only)
```

---

## 🔥 Shortcut: Transferring Data from Another System

Since embeddings extraction takes the longest time, you can save hours by copying files from a teammate's system. The critical files to transfer are:

| Priority | File | Size | What Happens If Missing |
|---|---|---|---|
| 🔴 **Must have** | `data/processed/embeddings.pt` | ~19 MB | Must regenerate (hours) |
| 🔴 **Must have** | `data/processed/ppi_graph.pt` | ~33 MB | Must regenerate |
| 🔴 **Must have** | `data/processed/ppi_graph_mapping.pt` | ~340 KB | Must regenerate |
| 🟡 **Recommended** | `data/processed/train.csv` | ~5.3 MB | Must run preprocess again |
| 🟡 **Recommended** | `data/processed/val.csv` | ~670 KB | Must run preprocess again |
| 🟡 **Recommended** | `data/processed/test.csv` | ~670 KB | Must run preprocess again |
| 🟢 **Optional** | `models/*.pth`, `models/*.pkl`, `models/graph_calibrator.json` | ~33 MB (+ ~200 MB RF baseline) | Can retrain (faster step) |

> [!TIP]
> The processed data and models are small enough (well under 1 GB without the RF baseline) to copy as a zip from a teammate's machine; extract it so the files land in `data/processed/` and `models/`.
> Then you can skip straight to **Step 3** (training) or even **Step 4** (web app) if models are included.

---

## ❓ Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure virtual environment is activated and `pip install -r requirements.txt` was run |
| `CUDA out of memory` | Reduce `batch_size` (e.g., `--batch_size 8`) or use CPU |
| `torch_geometric` install fails | Install it separately: `pip install torch-geometric` (may need specific version matching your PyTorch/CUDA) |
| `embeddings.pt not found` | Run `python src/data/feature_extraction.py` or copy from teammate |
| `train.csv not found` | Run `python src/data/preprocess_data.py` first |
| STRING DB download fails | Download manually from the URLs above and place in `data/raw/` |
| Frontend won't start | Run `npm install` in `app/frontend/` first |
