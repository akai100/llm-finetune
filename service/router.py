from fastapi import APIRouter, HTTPException
import asyncio
from service.gpu.schemas import InferenceRequest
from service.app import router as gpu_router, queues

router = APIRouter()

@router.post("/chat")
async def chat(req: dict):
    session_id = req["session_id"]
    query = req["query"]

    session = session_mgr.get(session_id)
    
    gpu = gpu_router.select_gpu_for_session(session)
    
    if gpu is None:
        raise HTTPException(status_code=503, detail="No GPU avaliable")

    if session is None:
        session = session_mgr.create(session_id, gpu.gpu_id)

    worker = workers[gpu.gpu_id]

    answer = await worker.generate_with_cache(
        prompt=query,
        gen_cfg={},
        session_state=session
    )

    return {"answer": answer}
