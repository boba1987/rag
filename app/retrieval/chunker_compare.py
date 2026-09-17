from __future__ import annotations

from app.models.schemas import RetrievedChunk


def pack_hits(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "section": chunk.section,
            "parent_id": chunk.parent_id,
            "score": chunk.score,
        }
        for chunk in chunks
    ]


def compare_chunkers(query: str, retrievers: dict) -> dict[str, list[dict]]:
    """Run the same query on each chunker collection. Retrievers must share retrieval settings."""
    return {name: pack_hits(retriever.search(query, infer=False)) for name, retriever in retrievers.items()}
