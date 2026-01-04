# training/checkpoint_guard.py
import torch
import os


class CheckpointGuard:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = cfg.training.output_dir
        os.makedirs(self.dir, exist_ok=True)

    def save_and_validate(self, model, optimizer, step):
        path = os.path.join(self.dir, f"ckpt-{step}.pt")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
            path,
        )

        self._validate(path)

    def _validate(self, path):
        ckpt = torch.load(path, map_location="cpu")

        for k, v in ckpt["model"].items():
            if torch.isnan(v).any():
                raise RuntimeError(f"NaN detected in checkpoint {path}")
