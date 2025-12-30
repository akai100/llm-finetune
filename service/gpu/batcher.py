import asyncio
import time

class DynamicBatcher:
    def __init__(self, max_batch_size=8, max_wait_ms=20):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def collect(self, queue: asyncio.Queue):
        batch = []
        start = time.time()

        while len(batch) < self.max_batch_size:
            timeout = self.max_wait_ms / 1000 - (time.time() - start)
            if timeout <= 0:
                break

            try:
                req = await asyncio.wait_for(queue.get(), timeout)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch
