from transformers import TrainerCallback
import torch
import logging

logger = logging.getLogger(__name__)

class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        if total_norm > args.max_grad_norm * 10:
            logger.error(f"Gradient explosion detected: {total_norm}")
            raise RuntimeError("Gradient explosion")
