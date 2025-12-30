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

num_gpus = torch.cuda.device_count()
gpu_states = [GPUState(i) for i in range(num_gpus)]
router = GPURouter(gpu_states)
session_mgr = SessionManager()
queues = [asyncio.Queue() for _ in range(num_gpus)]
sems = [asyncio.Semaphore(1) for _ in range(num_gpus)]
model = ModelService("gpt2")

workers = [
    GPUWorker(gpu_states[i], model, sems[i])
    for i in range(num_gpus)
]

@app.on_event("startup")
async def startup_event():
    for i in range(num_gpus):
        asyncio.create_task(workers[i].run(queues[i]))

@app.post("/chat")
async def chat(req: dict):
    session_id = req["session_id"]
    prompt = req["query"]

    session = session_mgr.get(session_id)
    gpu = router.select_gpu_for_session(session)
    if not gpu:
        raise HTTPException(503, "No GPU available")

    if not session:
        session = session_mgr.create(session_id, gpu.gpu_id)

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    queues[gpu.gpu_id].put_nowait(
        type("Req", (), {
            "prompt": prompt,
            "future": future
        })()
    )

    return {"answer": await future}
