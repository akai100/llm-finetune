import asyncio
import torch
from fastapi import FastAPI
from service.gpu.queue import InferenceQueue
from service.gpu.worker import GPUWorker
from service.gpu.semaphore import GPUSemaphore
from service.gpu.state import GPUState
from service.gpu.router import GPURouter
from service.model_loader import ModelService

app = FastAPI()

gpu_states = []
queues = {}
workers = {}

num_gpus = torch.cuda.device_count()

for gpu_id in range(num_gpus):
    state = GPUState(gpu_id, max_concurrent=1)
    gpu_states.append(state)

router = GPURouter(gpu_states)

@app.on_event("startup")
async def startup_event():
    for state in gpu_states:
        queue = InferenceQueue(max_size=50)
        model = ModelService("outputs/checkpoints/best")
        sem = GPUSemaphore(max_concurrent=1)
        worker = GPUWorker(state, model, sem)

        queues[state.gpu_id] = queue
        workers[state.gpu_id] = worker

        asyncio.create_task(worker.run(queue))
