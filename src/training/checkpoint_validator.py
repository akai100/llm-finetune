import torch
import os

def validate_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu")
        if not state:
            raise ValueError("Empty checkpoint")
    except Exception as e:
        raise RuntimeError(f"Invalid checkpoint: {path}") from e
