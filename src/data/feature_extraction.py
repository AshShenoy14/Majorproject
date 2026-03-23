import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import sys
import os
import gzip
from Bio import SeqIO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR

# ESM2 max context window — prevents OOM on long sequences
MAX_SEQ_LENGTH = 1022


class ESMFeatureExtractor:
    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D", device: str = "cpu"):
        self.device = device
        print(f"Loading ESM-2 model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_embeddings(
        self,
        sequences: Dict[str, str],
        batch_size: int = 8,
        save_path: str = None,
        save_every: int = 50,
        existing: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates embeddings for a dictionary of sequences.

        Args:
            sequences: Dict mapping ProteinID -> amino acid sequence.
            batch_size: Number of sequences per forward pass.
            save_path: If set, incrementally save embeddings every `save_every` batches.
            save_every: How often (in batches) to checkpoint to disk.
            existing: Previously computed embeddings to skip (resume support).

        Returns:
            Dictionary mapping ProteinID -> Embedding Tensor (float16).
        """
        embeddings = dict(existing) if existing else {}

        # Filter out already-processed proteins
        ids_all = list(sequences.keys())
        seqs_all = list(sequences.values())

        todo_ids, todo_seqs = [], []
        for pid, seq in zip(ids_all, seqs_all):
            if pid not in embeddings:
                todo_ids.append(pid)
                todo_seqs.append(seq)

        if len(todo_ids) < len(ids_all):
            print(f"  Resuming: {len(ids_all) - len(todo_ids)} already done, "
                  f"{len(todo_ids)} remaining")

        if not todo_ids:
            print("  All embeddings already computed!")
            return embeddings

        total_batches = (len(todo_seqs) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(0, len(todo_seqs), batch_size),
                              desc="Extracting features",
                              total=total_batches):
            batch_ids = todo_ids[batch_idx:batch_idx + batch_size]
            batch_seqs = todo_seqs[batch_idx:batch_idx + batch_size]

            try:
                self._embed_batch(batch_ids, batch_seqs, embeddings)
            except torch.cuda.OutOfMemoryError:
                # OOM fallback: process one-by-one
                torch.cuda.empty_cache()
                print(f"\n  ⚠ OOM on batch — retrying {len(batch_ids)} seqs one-by-one")
                for single_id, single_seq in zip(batch_ids, batch_seqs):
                    if single_id in embeddings:
                        continue
                    try:
                        self._embed_batch([single_id], [single_seq], embeddings)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"    ✗ Skipping {single_id} (seq len {len(single_seq)}) — OOM even solo")

            # Clear CUDA cache periodically to prevent fragmentation
            if self.device != "cpu" and (batch_idx // batch_size) % 10 == 0:
                torch.cuda.empty_cache()

            # Incremental save
            if save_path and ((batch_idx // batch_size + 1) % save_every == 0):
                torch.save(embeddings, save_path)
                tqdm.write(f"  💾 Checkpoint: {len(embeddings)} embeddings saved")

        return embeddings

    def _embed_batch(self, batch_ids, batch_seqs, embeddings):
        """Tokenize, run model, store results."""
        inputs = self.tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over sequence length (matches PPIDataset processing)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

        for pid, emb in zip(batch_ids, batch_embeddings):
            embeddings[pid] = emb.cpu().half()


if __name__ == "__main__":
    import numpy as np

    # 1. Load all proteins needed
    print("Loading datasets...")
    proteins = set()
    for split in ["train", "val", "test"]:
        path = PROCESSED_DATA_DIR / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            proteins.update(df["protein1"].unique())
            proteins.update(df["protein2"].unique())

    print(f"Total unique proteins: {len(proteins)}")

    # 2. Get Sequences
    from src.utils.paths import STRING_SEQUENCES_FILE

    print(f"Loading sequences from {STRING_SEQUENCES_FILE}...")
    sequences = {}

    if not STRING_SEQUENCES_FILE.exists():
        print(f"Error: Sequences file not found at {STRING_SEQUENCES_FILE}")
        print("Falling back to dummy sequences (NOT RECOMMENDED for production).")
        for p in proteins:
            length = np.random.randint(50, 100)
            aa = "ACDEFGHIKLMNPQRSTVWY"
            seq = "".join(np.random.choice(list(aa), length))
            sequences[p] = seq
    else:
        with gzip.open(STRING_SEQUENCES_FILE, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                full_id = record.id
                if full_id.startswith("9606."):
                    protein_id = full_id[5:]
                else:
                    protein_id = full_id

                if protein_id in proteins:
                    sequences[protein_id] = str(record.seq)

    print(f"Loaded {len(sequences)} sequences for {len(proteins)} required proteins.")

    # Sequence length stats
    lengths = [len(s) for s in sequences.values()]
    print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}, "
          f">{MAX_SEQ_LENGTH}: {sum(1 for l in lengths if l > MAX_SEQ_LENGTH)}")

    # Check coverage
    missing = len(proteins) - len(sequences)
    if missing > 0:
        print(f"Warning: {missing} proteins missing sequences. Filling with dummy.")
        for p in proteins:
            if p not in sequences:
                length = np.random.randint(50, 100)
                aa = "ACDEFGHIKLMNPQRSTVWY"
                seq = "".join(np.random.choice(list(aa), length))
                sequences[p] = seq

    # 3. Load existing embeddings (resume support)
    out_path = PROCESSED_DATA_DIR / "embeddings.pt"
    existing = {}
    if out_path.exists():
        try:
            existing = torch.load(out_path, weights_only=False)
            print(f"Loaded {len(existing)} existing embeddings (will skip these)")
        except Exception:
            print("Could not load existing embeddings — starting fresh")

    # 4. Extract Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ESMFeatureExtractor(device=device)

    # Sort sequences by length (ascending) so short ones batch together
    # This maximizes GPU efficiency and minimizes padding waste
    sorted_seqs = dict(sorted(sequences.items(), key=lambda x: len(x[1])))

    embeddings = extractor.get_embeddings(
        sorted_seqs,
        batch_size=16,       # Safer than 32 for 4GB VRAM
        save_path=str(out_path),
        save_every=50,       # Checkpoint every 50 batches
        existing=existing,
    )

    # 5. Final save
    torch.save(embeddings, out_path)
    print(f"\n✅ Done! {len(embeddings)} embeddings saved to {out_path}")
    print(f"   File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
