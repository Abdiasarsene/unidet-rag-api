
import logging

from fastapi import APIRouter, HTTPException

from schemas.chat import ChatRequest, ChatResponse, Source
from services.retrieval import retrieve_context
from services.llm import generate_answer


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        chunks = retrieve_context(request.question)
        answer = generate_answer(request.question, chunks)

        sources = []
        seen_sources = set()

        for chunk in chunks:
            title = chunk["title"]
            url = chunk.get("url")

            # Déduplique par URL si disponible, sinon par titre.
            source_key = url or title

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)
            sources.append(
                Source(
                    title=title,
                    url=url,
                )
            )

        return ChatResponse(
            answer=answer,
            sources=sources,
        )

    except Exception:
        logger.exception("Échec du traitement de la demande de chat.")

        raise HTTPException(
            status_code=503,
            detail=(
                "Le service de l'assistant est temporairement "
                "indisponible. Veuillez réessayer plus tard."
            ),
        ) from None
