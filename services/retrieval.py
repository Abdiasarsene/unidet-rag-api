
import cohere

from db.database import search_similar_chunks
from utils.config import settings


cohere_client = cohere.ClientV2(
    api_key=settings.cohere_api_key.get_secret_value()
)


def embed_query(question: str) -> list[float]:
    """Transforme une question en embedding Cohere."""

    question = question.strip()
    if not question:
        raise ValueError("La question ne peut pas être vide.")

    response = cohere_client.embed(
        model=settings.cohere_embed_model,
        texts=[question],
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=settings.cohere_embed_dimension,
    )

    embeddings = response.embeddings.float
    if not embeddings:
        raise RuntimeError("Cohere n'a renvoyé aucun embedding.")

    return list(embeddings[0])


def retrieve_context(question: str) -> list[dict]:
    """Retourne les passages pertinents pour une question."""

    embedding = embed_query(question)
    chunks = search_similar_chunks(embedding)

    results = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}

        if not isinstance(metadata, dict):
            metadata = {}

        title = (
            metadata.get("title")
            or chunk.get("source_name")
            or chunk.get("chunk_name")
            or "Document UNIDET"
        )

        url = metadata.get("url") or metadata.get("source_url")

        results.append(
            {
                "content": chunk["content"],
                "title": title,
                "url": url,
                "similarity": float(chunk["similarity"]),
            }
        )

    return results