# training/step_runner.py
import torch


class StepRunner:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def run_step(self, batch):
        outputs = self.model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf loss detected")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.stability.max_grad_norm
        )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()
