# Archived experiments (not part of the reported results)

- `hetero_graph_model.py`, `train_hetero_graph_model.py`, `hetero_graph_construction.py`: a heterogeneous-graph GNN
  pipeline that was explored but is **not used** in the final reported results.
- `metric_optimization/`: stage-wise meta-learner hyper-parameter / topology-feature search scripts written for the
  earlier 8-feature ensemble and 5-epoch OOF folds. They import APIs that no longer exist (`train_fold_sequence`,
  `train_fold_gat`, `bio_features`) and are kept for provenance only; do not use them for reported numbers.
