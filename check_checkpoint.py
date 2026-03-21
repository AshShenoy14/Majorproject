import torch
import os
checkpoint_path = r'e:\majorproject\checkpoints\graph_checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Patience: {checkpoint.get('epochs_no_improve', 'N/A')}")
else:
    print("Checkpoint not found.")
