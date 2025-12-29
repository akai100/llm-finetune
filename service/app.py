from fastapi import FastAPI
from service.router import router

app = FastAPI(title="Enterprise LLM Service")
app.include_router(router)

