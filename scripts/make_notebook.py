import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TransGraph-PPI: Complete Model Training & Evaluation Pipeline\n",
                "### Model Architecture: ESM-2 150M (`facebook/esm2_t30_150M_UR50D`) + GraphSAGE + XGBoost Stacking\n",
                "\n",
                "This notebook executes the complete end-to-end training and benchmark pipeline on a T4 GPU:\n",
                "1. **Environment Setup & GPU Check**\n",
                "2. **Dependency & Torchvision Fix** (uninstalls conflicting torchvision to load EsmModel cleanly)\n",
                "3. **Dataset Splits Setup** (restores or uploads `train.csv`, `val.csv`, `test.csv`)\n",
                "4. **Dataset Verification** (verifies pair-disjoint split integrity)\n",
                "5. **ESM-2 150M Feature Extraction** (smart skip: restores pre-computed `embeddings.pt` from Drive in seconds!)\n",
                "6. **Graph Construction** (smart skip: restores `ppi_graph.pt` from Drive in seconds!)\n",
                "7. **Sequence Model Training** (Residual MLP on 640-d embeddings with early stopping)\n",
                "8. **Graph Model Training** (GraphSAGE link prediction + Platt calibration)\n",
                "9. **Random Forest Baseline Training**\n",
                "10. **5-Fold OOF XGBoost Ensemble Training**\n",
                "11. **Final Test Evaluation & Comparison** (`final_test_metrics.json`)\n",
                "12. **External Benchmarks** (Cold-Start, SHS27k, HuRI)\n",
                "13. **Artifact Packaging & Checksums** (`esm150m_results.zip`)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Verify GPU Acceleration (T4 GPU)\n",
                "import torch\n",
                "print(\"PyTorch Version:\", torch.__version__)\n",
                "if torch.cuda.is_available():\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
                "    print(f\"CUDA is ACTIVE! GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)\")\n",
                "else:\n",
                "    print(\"WARNING: CUDA is not active. If in Colab, select Runtime -> Change runtime type -> T4 GPU.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Environment Setup & Google Drive Mount (if on Google Colab)\n",
                "import os\n",
                "import sys\n",
                "\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    if not os.path.exists(\"/content/drive/MyDrive\"):\n",
                "        drive.mount(\"/content/drive\")\n",
                "        print(\"Google Drive mounted.\")\n",
                "    if os.path.exists(\"/content\") and not os.path.exists(\"/content/Majorproject\"):\n",
                "        !git clone -b metric-optimization https://github.com/AshShenoy14/Majorproject.git /content/Majorproject\n",
                "        %cd /content/Majorproject\n",
                "    elif os.path.exists(\"/content/Majorproject\"):\n",
                "        %cd /content/Majorproject\n",
                "except ImportError:\n",
                "    print(\"Running in local VS Code / Colab Extension environment.\")\n",
                "\n",
                "print(\"Working directory:\", os.getcwd())\n",
                "sys.path.append(os.getcwd())\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Fix Dependencies & Eliminate Torchvision Conflicts\n",
                "# Uninstall torchvision because it triggers C++ operator conflicts with PyTorch on Colab\n",
                "!pip uninstall -y torchvision\n",
                "!pip install -q -r requirements.txt\n",
                "print(\"Requirements installed successfully. Torchvision removed (clean NLP/graph runtime).\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3.5: Load / Restore Dataset Splits (train.csv, val.csv, test.csv)\n",
                "import os\n",
                "\n",
                "os.makedirs(\"data/processed\", exist_ok=True)\n",
                "drive_data = \"/content/drive/MyDrive/transgraph_ppi_esm150m/data_processed\"\n",
                "\n",
                "# 1. Try restoring from Google Drive backup if available\n",
                "if os.path.exists(f\"{drive_data}/train.csv\"):\n",
                "    print(\"Restoring train.csv, val.csv, test.csv from Google Drive...\")\n",
                "    !cp -r \"$drive_data\"/*.csv data/processed/ 2>/dev/null || true\n",
                "    print(\"Splits restored from Google Drive!\")\n",
                "\n",
                "# 2. If not on Drive, prompt for quick upload or generate\n",
                "if not os.path.exists(\"data/processed/train.csv\"):\n",
                "    print(\"Notice: train.csv not found in data/processed/.\")\n",
                "    try:\n",
                "        from google.colab import files\n",
                "        print(\"Please select train.csv, val.csv, test.csv, bio_metadata_cache.csv from your local E:\\\\majorproject\\\\data\\\\processed folder:\")\n",
                "        uploaded = files.upload()\n",
                "        !mv *.csv data/processed/ 2>/dev/null || true\n",
                "    except Exception:\n",
                "        print(\"Generating splits using preprocess_data.py...\")\n",
                "        !python src/data/preprocess_data.py --seed 42\n",
                "\n",
                "!ls -lh data/processed/*.csv\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 4: Verify Dataset Splits & Negative Sampling\n",
                "!python src/data/collect_ppi.py\n",
                "!python scripts/verify_splits.py\n",
                "!python scripts/audit_measurements.py --out assets/evaluation/audit/audit_after_data.json --skip-model-metrics\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: ESM-2 150M Feature Extraction (Smart Check)\n",
                "# Checks if embeddings exist locally or in Drive backup to avoid re-extracting\n",
                "import os\n",
                "\n",
                "emb_path = \"data/processed/embeddings.pt\"\n",
                "drive_emb = \"/content/drive/MyDrive/transgraph_ppi_esm150m/data_processed/embeddings.pt\"\n",
                "\n",
                "if os.path.exists(emb_path) and os.path.getsize(emb_path) > 1000000:\n",
                "    size_mb = os.path.getsize(emb_path) / (1024 * 1024)\n",
                "    print(f\"Embeddings already exist at {emb_path} ({size_mb:.1f} MB). Skipping extraction.\")\n",
                "elif os.path.exists(drive_emb):\n",
                "    print(f\"Restoring pre-computed embeddings from Google Drive: {drive_emb}...\")\n",
                "    !mkdir -p data/processed\n",
                "    !cp \"$drive_emb\" \"$emb_path\"\n",
                "    print(\"Embeddings successfully restored from Drive!\")\n",
                "else:\n",
                "    print(\"Extracting 640-d ESM-2 150M embeddings for all 12,323 proteins (takes ~25-30 mins on T4 GPU)...\")\n",
                "    !python src/data/feature_extraction.py --batch-size 16\n",
                "    print(\"Feature extraction complete.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 6: Graph Construction (640-d ESM + 3 Topological features = 643-d nodes)\n",
                "import os\n",
                "\n",
                "graph_path = \"data/processed/ppi_graph.pt\"\n",
                "drive_graph = \"/content/drive/MyDrive/transgraph_ppi_esm150m/data_processed/ppi_graph.pt\"\n",
                "drive_map = \"/content/drive/MyDrive/transgraph_ppi_esm150m/data_processed/ppi_graph_mapping.pt\"\n",
                "\n",
                "if os.path.exists(graph_path) and os.path.getsize(graph_path) > 1000000:\n",
                "    print(f\"PPI graph already exists at {graph_path}. Skipping.\")\n",
                "elif os.path.exists(drive_graph) and os.path.exists(drive_map):\n",
                "    print(\"Restoring pre-built graph from Google Drive...\")\n",
                "    !mkdir -p data/processed\n",
                "    !cp \"$drive_graph\" \"$graph_path\"\n",
                "    !cp \"$drive_map\" \"data/processed/ppi_graph_mapping.pt\"\n",
                "    print(\"Graph restored successfully from Drive.\")\n",
                "else:\n",
                "    print(\"Building interaction graph and calculating topological features...\")\n",
                "    !python src/data/graph_construction.py\n",
                "\n",
                "# Backup processed data to Drive if available\n",
                "if os.path.exists(\"/content/drive/MyDrive\"):\n",
                "    !mkdir -p /content/drive/MyDrive/transgraph_ppi_esm150m/data_processed\n",
                "    !cp -r data/processed/*.csv data/processed/*.pt /content/drive/MyDrive/transgraph_ppi_esm150m/data_processed/ 2>/dev/null || true\n",
                "    print(\"Data backed up to Google Drive.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 7: Train Sequence Model (ESM-2 150M Residual MLP)\n",
                "import os\n",
                "checkpoint_dir = \"/content/drive/MyDrive/transgraph_ppi_esm150m/checkpoints\" if os.path.exists(\"/content/drive/MyDrive\") else \"checkpoints\"\n",
                "os.makedirs(checkpoint_dir, exist_ok=True)\n",
                "os.makedirs(\"models\", exist_ok=True)\n",
                "\n",
                "print(\"Training Sequence Model with early stopping (patience 10)...\")\n",
                "!python src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt --checkpoint_dir \"{checkpoint_dir}\" --ckpt_every 5\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 8: Train Graph Model (GraphSAGE Link Prediction & Platt Calibration)\n",
                "print(\"Training GraphSAGE Model and calculating Platt probability calibration...\")\n",
                "!python src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt --checkpoint_dir \"{checkpoint_dir}\" --ckpt_every 5\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 9: Train Random Forest Baseline Model\n",
                "print(\"Fitting Random Forest Baseline on train.csv...\")\n",
                "!python src/training/train_random_forest.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 10: Train 5-Fold OOF Stacking Ensemble (XGBoost)\n",
                "print(\"Training 5-Fold Out-Of-Fold base models and XGBoost Stacking Ensemble...\")\n",
                "!python src/training/train_ensemble.py --checkpoint_dir \"{checkpoint_dir}\" --ckpt_every 5\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 11: Comprehensive Held-Out Test Evaluation\n",
                "# Evaluates all models on test.csv (20,172 pairs) and saves final_test_metrics.json\n",
                "print(\"Evaluating models on held-out test split...\")\n",
                "!python src/analysis/compare_models.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 12: External Benchmarks & Generalization Evaluation\n",
                "print(\"--- 1. Cold-Start Evaluation ---\")\n",
                "!python scripts/cold_start_eval.py\n",
                "\n",
                "print(\"\\n--- 2. SHS27k External Benchmark ---\")\n",
                "!python scripts/external_benchmark_shs27k.py\n",
                "\n",
                "print(\"\\n--- 3. HuRI Independent Benchmark ---\")\n",
                "!python scripts/external_benchmark_huri.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 13: Display Metrics & Generate Audit Hashes\n",
                "import json\n",
                "import pandas as pd\n",
                "\n",
                "metrics_file = \"assets/evaluation/final_test_metrics.json\"\n",
                "if os.path.exists(metrics_file):\n",
                "    with open(metrics_file, \"r\") as f:\n",
                "        metrics_data = json.load(f)\n",
                "    print(\"\\n=================== FINAL TEST PERFORMANCE SUMMARY ===================\")\n",
                "    summary_df = pd.DataFrame(metrics_data).T\n",
                "    cols = [c for c in [\"accuracy\", \"precision\", \"recall\", \"f1\", \"roc_auc\", \"pr_auc\"] if c in summary_df.columns]\n",
                "    print(summary_df[cols])\n",
                "\n",
                "# Hashes and audits\n",
                "!sha256sum data/processed/*.csv data/processed/*.pt models/*.pth models/*.pkl models/*.json > assets/evaluation/artifact_hashes.txt 2>/dev/null || true\n",
                "!python scripts/audit_measurements.py --out assets/evaluation/audit/audit_after_full.json\n",
                "!pip freeze > requirements-lock.txt\n",
                "print(\"Artifact hashes, audits, and requirements-lock.txt updated.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 14: Package Trained Models & Artifacts\n",
                "zip_target = \"/content/drive/MyDrive/esm150m_results.zip\" if os.path.exists(\"/content/drive/MyDrive\") else \"esm150m_results.zip\"\n",
                "\n",
                "!zip -r \"{zip_target}\" \\\n",
                "    models \\\n",
                "    assets/evaluation \\\n",
                "    data/processed/embeddings.pt \\\n",
                "    data/processed/ppi_graph.pt \\\n",
                "    data/processed/ppi_graph_mapping.pt \\\n",
                "    requirements-lock.txt\n",
                "\n",
                "print(f\"\\nSUCCESS: All models and metrics packaged into: {zip_target}\")\n"
            ]
        }
    ]
}

with open("TransGraph_PPI_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Updated TransGraph_PPI_Training.ipynb generated successfully!")
