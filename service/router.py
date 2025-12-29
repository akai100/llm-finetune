from fastapi import APIRouter
from rag.retriever import Retriever
from rag.prompt import build_rag_prompt
from service.model_loader import ModelService

router = APIRouter()
retriever = Retriever("rag/index/faiss.index")
model = ModelService("outputs/checkpoints/best")

@router.post("/chat")
def chat(req: dict):
    query = req["query"]
    doc_ids = retriever.retrieve(query)
    docs = [f"Doc {i}" for i in doc_ids]  # placeholder

    prompt = build_rag_prompt(query, docs)
    answer = model.generate(prompt, gen_cfg={})
    return {"answer": answer}

