import threading

class GPUState:
    def __init__(self, gpu_id: int, max_concurrent: int):
        self.gpu_id = gpu_id
        self.active = 0
        self.lock = threading.lock()
        self.max_concurrent = max_concurrent
        self.healthy = True

        # 初始化显存信息
        props = torch.cuda.get_device_properties(gpu_id)
        self.total_memory = props.total_memory  # bytes

    def refresh_memory(self):
        torch.cuda.set_device(self.gpu_id)
        free, total = torch.cuda.mem_get_info()
        return free, total

    def can_accept(self):
        if not self.healthy:
            return False

        free, total = self.refresh_memory()
        free_ratio = free / total

        with self.lock:
            return (
                free_ratio >= min_free_ratio and
                self.active < self.max_concurrent
            )

    def on_start(self):
        with self.lock:
            self.active += 1

    def on_finish(self):
        with self.lock:
            self.active -= 1

    def mark_unhealthy(self):
        with self.lock:
          self.healthy = False
  
