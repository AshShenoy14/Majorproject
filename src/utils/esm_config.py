"""
Single source of truth for the ESM-2 protein language model used across the project.

Every embedding (data/processed/embeddings.pt, the graph node features in ppi_graph.pt, and the embeddings the
backend computes live for query proteins) must come from this model, and every SequencePPIModel must be built with
ESM_EMBED_DIM. Changing ESM_MODEL_NAME means regenerating embeddings and retraining every model.
"""

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"  # 150M parameters, 30 layers
ESM_EMBED_DIM = 640
