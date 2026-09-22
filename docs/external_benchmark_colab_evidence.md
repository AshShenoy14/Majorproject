# External benchmark run evidence (Colab T4, console excerpt)

Console output of the Colab T4 GPU run of `scripts/external_benchmark_shs27k.py` and
`scripts/external_benchmark_huri.py` against the same checkpoints behind the main 92.13% result. This
is a transcription of pasted console output, not a machine-generated log file. The printed JSON below
is byte-identical to the committed `assets/evaluation/external_benchmark_shs27k.json` and
`external_benchmark_huri.json` (commit `9995876`).

## SHS27k
```
Using device: cuda
Loaded SHS27k: 26944 rows
Unique sequences: 1690, unique canonical positive pairs: 7624
Generated 7624 negative pairs (target 7624)
Matched to existing training protein (byte-identical sequence, with embedding): 1375; novel: 315
Extracting ESM-2 embeddings for 315 novel sequences on cuda...
Benchmark set: 15248 pairs (0.500 positive rate)
Loaded meta-learner from /content/Majorproject/models/ensemble_model.pkl
Loaded GraphSAGE calibrator from /content/Majorproject/models/graph_calibrator.json
Scoring pairs... (0/15248 .. 15000/15248)

overall:                          acc 0.6979  roc_auc 0.8023  f1 0.6119  n 15248
both_proteins_seen_in_training:   acc 0.7057  roc_auc 0.8050  f1 0.6590  n 10840
at_least_one_novel_protein:       acc 0.6788  roc_auc 0.7900  f1 0.4363  n 4408
```

## HuRI / HI-union
```
Using device: cuda
Full HI-union: 64006 pairs, 9094 unique genes
Sampled 700 unique positive gene pairs
1028 gene sequences already cached; fetching 3 new ones from Ensembl REST...
  batch 0-3 failed x3 (404 Not Found) -> WARNING: 1 batch skipped, genes simply absent, not mislabeled
Genes with a usable sequence: 1028; positive pairs retained: 697
Generated 697 negative pairs
Matched to existing training protein (byte-identical sequence): 381; novel: 647
Extracting ESM-2 embeddings for 647 novel sequences on cuda...
Benchmark set: 1394 pairs (0.500 positive rate)
Loaded meta-learner from /content/Majorproject/models/ensemble_model.pkl
Loaded GraphSAGE calibrator from /content/Majorproject/models/graph_calibrator.json
Scoring pairs... (0/1394 .. 1000/1394)

overall:                          acc 0.5230  roc_auc 0.5726  f1 0.1441  n 1394
  predicted_positive_rate 0.0574 (vs true 0.500) | confusion_matrix tn=673 fp=24 fn=641 tp=56
both_proteins_seen_in_training:   acc 0.6205  roc_auc 0.6358  f1 0.3368  n 166
at_least_one_novel_protein:       acc 0.5098  roc_auc 0.5665  f1 0.1173  n 1228
```

## Test suite on Colab
```
tests/test_external_benchmarks.py ..                                     [100%]
============================== 2 passed in 0.02s ===============================
```

## Notes
- The single failed Ensembl REST batch (3 genes, HTTP 404) is expected, retry-and-skip behavior
  (`scripts/external_benchmark_huri.py`'s `fetch_sequences`), not a script bug: those 3 genes are
  simply absent from the cache and their pairs were dropped (700 sampled -> 697 retained), which is
  why `n_pairs_sampled_from_full_huri` (700) and `n_total_pairs_scored` (1394 = 697*2) differ.
- No retraining occurred in either run; only inference against `models/sequence_model_best.pth`,
  `models/graph_model_best.pth`, `models/graph_calibrator.json` and `models/ensemble_model.pkl`,
  loaded exactly as `app/backend/main.py` loads them for `/predict`.
