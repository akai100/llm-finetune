# training/pruning/pruner.py
import torch
import torch.nn.utils.prune as prune


class StructuredPruner:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.pruned = False

    def apply(self):
        for name, module in self.model.named_modules():
            if any(k in name for k in self.cfg.pruning.target_modules):
                if hasattr(module, "weight"):
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=self.cfg.pruning.sparsity,
                        n=2,
                        dim=0,
                    )
        self.pruned = True

    def remove(self):
        for module in self.model.modules():
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")
