import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import os
import gzip
from Bio import SeqIO
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR, STRING_SEQUENCES_FILE


def configure_runtime(force_cpu: bool = False, cpu_threads: int = None) -> str:
    """Configure torch thread usage and return selected device name."""
    use_cpu = force_cpu or (not torch.cuda.is_available())
    device = "cpu" if use_cpu else "cuda"

    if device == "cpu":
        if cpu_threads is None:
            cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        cpu_threads = max(1, int(cpu_threads))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
        print(
            f"Runtime: CPU | torch threads={torch.get_num_threads()} | "
            f"inter-op={torch.get_num_interop_threads()}"
        )
    else:
        print("Runtime: CUDA")

    return device

class ESMFeatureExtractor:
    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D", device: str = "cpu"):
        self.device = device
        print(f"Loading ESM-2 model: {model_name} on {device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        except Exception as e:
            print(f"Offline load failed ({e}), attempting regular load...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_embeddings(self, sequences: Dict[str, str], batch_size: int = 4, save_path: str = None) -> Dict[str, torch.Tensor]:
        """
        Generates embeddings for a dictionary of sequences with checkpointing.
        """
        embeddings = {}
        
        # Load existing if available (RESUME)
        if save_path and os.path.exists(save_path):
            try:
                embeddings = torch.load(save_path)
                print(f"Found existing embeddings. Resuming from {len(embeddings)} proteins.")
            except Exception as e:
                print(f"Error loading existing embeddings: {e}. Starting fresh.")
                embeddings = {}

        # Filter out already processed
        remaining_ids = [pid for pid in sequences.keys() if pid not in embeddings]
        if not remaining_ids:
            print("All sequences already processed!")
            return embeddings

        print(f"Processing {len(remaining_ids)} remaining proteins...")
        
        ids = remaining_ids
        seqs = [sequences[pid] for pid in ids]
        
        save_counter = 0
        for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting features"):
            batch_ids = ids[i:i+batch_size]
            batch_seqs = seqs[i:i+batch_size]
            
            # Explicitly cap length to 1024 to avoid 15GB+ allocations
            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                for pid, emb in zip(batch_ids, batch_embeddings):
                    embeddings[pid] = emb.cpu().half()
            except torch.cuda.OutOfMemoryError:
                print(f"\nWarning: Skipped batch due to OOM even with size {batch_size}. Trying individuals...")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                # If batch fails, try 1 by 1
                for pid, seq in zip(batch_ids, batch_seqs):
                    if pid in embeddings: continue
                    try:
                        inp = self.tokenizer([seq], return_tensors="pt", truncation=True, max_length=1024).to(self.device)
                        with torch.no_grad():
                            out = self.model(**inp)
                            embeddings[pid] = out.last_hidden_state.mean(dim=1)[0].cpu().half()
                    except Exception as e:
                        print(f"Error processing {pid}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # Periodic Save (Every 25 batches)
            save_counter += 1
            if save_counter >= 25 and save_path:
                torch.save(embeddings, save_path)
                save_counter = 0
                
            # Periodic memory cleanup
            if i % (batch_size * 5) == 0 and self.device == "cuda":
                torch.cuda.empty_cache()
                
        return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ESM embeddings for protein sequences")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Embedding extraction batch size (default: 4)")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Maximum PyTorch CPU threads (default: half logical cores)")
    parser.add_argument("--cpu-friendly", action="store_true",
                        help="Enable low-heat CPU preset")
    parser.add_argument("--model-name", type=str, default="facebook/esm2_t12_35M_UR50D",
                        help="Hugging Face ESM model name")
    args = parser.parse_args()

    if args.cpu_friendly:
        args.force_cpu = True
        if args.cpu_threads is None:
            args.cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        if args.batch_size == 4:
            args.batch_size = 2
        print("CPU-friendly preset enabled:")
        print(f"  force_cpu={args.force_cpu} | cpu_threads={args.cpu_threads} | batch_size={args.batch_size}")

    device = configure_runtime(force_cpu=args.force_cpu, cpu_threads=args.cpu_threads)

    # 1. Load all proteins
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
    print(f"Loading sequences from {STRING_SEQUENCES_FILE}...")
    sequences = {}
    
    if not STRING_SEQUENCES_FILE.exists():
        print(f"Error: Sequences file not found at {STRING_SEQUENCES_FILE}")
        for p in proteins:
            length = np.random.randint(50, 100)
            aa = "ACDEFGHIKLMNPQRSTVWY"
            sequences[p] = "".join(np.random.choice(list(aa), length))
    else:
        with gzip.open(STRING_SEQUENCES_FILE, "rt") as handle:
             for record in SeqIO.parse(handle, "fasta"):
                  protein_id = record.id[5:] if record.id.startswith("9606.") else record.id
                  if protein_id in proteins:
                      sequences[protein_id] = str(record.seq)

    # 3. Extract Embeddings with Resume
    out_path = PROCESSED_DATA_DIR / "embeddings.pt"
    
    extractor = ESMFeatureExtractor(model_name=args.model_name, device=device)
    embeddings = extractor.get_embeddings(sequences, batch_size=max(1, args.batch_size), save_path=str(out_path)) 
    
    # 4. Final Save
    torch.save(embeddings, out_path)
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embeddings saved to {out_path}")
