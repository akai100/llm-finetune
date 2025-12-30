import random
from service.gpu.state import GPUState

class GPURouter:
    def __init__(self, gpu_states):
        self.gpu_states = gpu_states

    def select_gpu(self):
        # 过滤可用 GPU
        candidates = [
            g for g in self.gpu_states if g.can_accept()
        ]
        if not candidates:
            return None

        # 选择 active 最少的
        candidates.sort(key=lambda g: g.active)
        return candidates[0]
