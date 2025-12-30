from fastapi import APIRouter, HTTPException
import asyncio
from service.gpu.schemas import InferenceRequest
from service.app import router as gpu_router, queues

router = APIRouter()

@router.post("/chat")
async def chat(req: dict):
    gpu = gpu_router.select_gpu()
    if gpu is None:
        raise HTTPException(status_code=503, detail="All GPUs busy")

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    inf_req = InferenceRequest(
        prompt=req["query"],
        gen_cfg={},
        future=future
    )

    await queues[gpu.gpu_id].put(inf_req)

    try:
        result = await asyncio.wait_for(future, timeout=60)
        return {"answer": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
