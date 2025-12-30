import asyncio
import torch
import logging

logger = logging.getLogger(__name__)

class GPUWorker:
    def __init__(self, gpu_state, model_service, semaphore):
        self.gpu_state = gpu_state
        self.model = model_service
        self.sem = semaphore

    async def run(self, queue):
        torch.cuda.set_device(self.gpu_state.gpu_id)

        while True:
            req = await queue.get()
            await self.sem.acquire()
            self.gpu_state.on_start()

            try:
                result = await asyncio.to_thread(
                    self.model.generate,
                    req.prompt,
                    req.gen_cfg
                )
                req.future.set_result(result)
            except Exception as e:
                logger.exception(e)
                req.future.set_exception(e)
            finally:
                self.gpu_state.on_finish()
                self.sem.release()
                queue.queue.task_done()
