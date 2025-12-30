import threading

class GPUState:
    def __init__(self, gpu_id: int, max_concurrent: int):
        self.gpu_id = gpu_id
        self.active = 0
        self.lock = threading.lock()
        self.max_concurrent = max_concurrent
        self.healthy = True

    def can_accept(self):
        with self.lock:
            return self.healthy and self.active < self.max_concurrent

    def on_start(self):
        with self.lock:
            self.active += 1

    def on_finish(self):
        with self.lock:
            self.active -= 1

    def mark_unhealthy(self):
        with self.lock:
          self.healthy = False
  
