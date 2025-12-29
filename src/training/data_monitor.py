import logging
import torch

logger = logging.getLogger(__name__)

class DataDistributionMonitor:
    def __init__(self):
        self.token_lens = []

    def update(self, input_ids):
        self.token_lens.append(input_ids.size(1))

    def check(self, step):
        if step < 100:
            return
        avg_len = sum(self.token_lens[-100:]) / 100
        if avg_len < 20:
            logger.error(
                f"[DataDrift] Suspicious short sequences detected at step {step}"
            )
