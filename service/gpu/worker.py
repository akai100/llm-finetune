import asyncio
import torch
import logging

logger = logging.getLogger(__name__)

class GPUWorker:
    def __init__(self, gpu_state, model_service, semaphore):
        self.gpu_state = gpu_state
        self.model = model_service
        self.sem = semaphore
        self.batcher = DynamicBatcher(
            max_batch_size=8,
            max_wait_ms=20
        )

    async def generate_with_cache(
        self,
        prompt,
        gen_cfg,
        session_state
    ):
        torch.cuda.set_device(self.gpu_state.gpu_id)

        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        outputs = await asyncio.to_thread(
            self.model.model.generate,
            **inputs,
            past_key_values=session_state.past_key_values,
            use_cache=True,
            **gen_cfg
        )

        session_state.past_key_values = outputs.past_key_values

        return self.model.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )

    async def run(self, queue):
        torch.cuda.set_device(self.gpu_state.gpu_id)

        while True:
            batch = await self.batcher.collect(queue)
            if not batch:
                continue
            
            await self.sem.acquire()
            self.gpu_state.on_start()

            try:
                prompts = [r.prompt for r in batch]
                gen_cfg = batch[0].gen_cfg  # 简化：batch 内共享

                outputs = await asyncio.to_thread(
                    self.model.generate_batch,
                    prompts,
                    gen_cfg
                )

                for r, out in zip(batch, outputs):
                    r.future.set_result(out)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.gpu_state.mark_unhealthy()
                for r in batch:
                    r.future.set_exception(e)
            finally:
                self.gpu_state.on_finish()
                self.sem.release()
                for _ in batch:
                    queue.task_done()
