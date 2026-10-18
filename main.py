
from fastapi import FastAPI

from api.chat import router as chat_router


app = FastAPI(
    title="UNIDET API",
    description="API de l'assistant intelligent UNIDET",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


app.include_router(
    chat_router,
    prefix="/api/v1",
    tags=["Chat"],
)
