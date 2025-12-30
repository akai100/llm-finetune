import random
from service.gpu.state import GPUState

class GPURouter:
    def __init__(self,
                 gpu_states,
                 alpha=1.0,
                 beta=0.5,
                 gamma=0.2,
                 min_free_ratio=0.15):
        self.gpu_states = gpu_states
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_free_ratio = min_free_ratio

    def score(self, gpu, queue_length):
        free, total = gpu.refresh_memory()
        free_ratio = free / total

        return (
            self.alpha * free_ratio
            - self.beta * gpu.active
            - self.gamma * queue_length
        )

    def select_gpu(self):
        candidates = []
        for gpu in self.gpu_states:
            if not gpu.can_accept(self.min_free_ratio):
                continue

            q_len = queues[gpu.gpu_id].queue.qsize()
            s = self.score(gpu, q_len)
            candidates.append((s, gpu))

        if not candidates:
            return None

        # 选 score 最高的
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
