#!/usr/bin/env bash
# Full pipeline on a Google Colab GPU runtime (T4). Run from the repo root after:
#   from google.colab import drive; drive.mount('/content/drive')
#   !pip install -r requirements.txt   (install torch with the matching CUDA wheel first if needed)
# Checkpoints go to Drive so a disconnected session resumes: base models resume from the last saved epoch,
# and train_ensemble.py skips OOF folds that already finished. No hyper-parameter is tuned here.
#
# REUSE_SPLITS=1 keeps the existing data/processed/{train,val,test}.csv (upload them first; collect_ppi.py
# downloads any missing raw STRING file) so a model change such as the ESM-2 35M -> 150M switch is compared
# on exactly the same pairs. Leave it unset to re-download STRING and rebuild the splits from scratch.
set -euo pipefail
DRIVE=${DRIVE:-/content/drive/MyDrive/transgraph_ppi_esm150m}   # fresh folder: no checkpoints from the 35M run
CK="$DRIVE/checkpoints"; mkdir -p "$CK" models assets/evaluation

if [ "${REUSE_SPLITS:-0}" = "1" ]; then
  python src/data/collect_ppi.py   # only downloads files that are missing; the splits are not touched
else
  python src/data/collect_ppi.py
  python src/data/preprocess_data.py --seed 42
fi
python scripts/verify_splits.py
python scripts/audit_measurements.py --out assets/evaluation/audit/audit_after_data.json --skip-model-metrics

rm -f data/processed/embeddings.pt data/processed/ppi_graph.pt data/processed/ppi_graph_mapping.pt   # never mix embedding models
python src/data/feature_extraction.py --batch-size 16
python src/data/graph_construction.py
mkdir -p "$DRIVE/data_processed" && cp -r data/processed/*.csv data/processed/*.pt "$DRIVE/data_processed/"   # splits + embeddings + graph: needed to re-verify metrics later; /content is wiped on disconnect

python src/training/train_sequence_model.py --embedding_path data/processed/embeddings.pt --checkpoint_dir "$CK" --ckpt_every 5
python src/training/train_graph_model.py --graph_path data/processed/ppi_graph.pt --checkpoint_dir "$CK" --ckpt_every 5
python src/training/train_random_forest.py
python src/training/train_ensemble.py --checkpoint_dir "$CK" --ckpt_every 5
python src/analysis/compare_models.py
python scripts/cold_start_eval.py
python scripts/external_benchmark_shs27k.py
python scripts/external_benchmark_huri.py

sha256sum data/processed/*.csv data/processed/*.pt models/*.pth models/*.pkl models/*.json > assets/evaluation/artifact_hashes.txt   # provenance for later local re-verification
cp -r models assets/evaluation "$DRIVE/"                       # keep artefacts even if the session dies later (before the audit, which can fail)
python scripts/audit_measurements.py --out assets/evaluation/audit/audit_after_full.json
pip freeze > requirements-lock.txt                             # reproducibility: commit this next to requirements.txt
