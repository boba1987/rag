from __future__ import annotations

from app.config import DENSE_TOP_K, QDRANT_COLLECTION
from app.ingestion.embedder import Embedder, get_embedder
from app.ingestion.indexer import chunk_from_payload, get_qdrant_client
from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.retrieval.filters import apply_section_filter, build_qdrant_filter, infer_filters, merge_filters


class DenseRetriever:
    """Embed the query and return the top-k cosine neighbors from Qdrant."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        client=None,
        collection: str = QDRANT_COLLECTION,
        top_k: int = DENSE_TOP_K,
    ) -> None:
        self._embedder = embedder or get_embedder()
        self._client = client or get_qdrant_client()
        self._collection = collection
        self._top_k = top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        infer: bool = False,
    ) -> list[RetrievedChunk]:
        limit = top_k or self._top_k
        applied = merge_filters(filters, infer_filters(query)) if infer else filters
        vector = self._embedder.embed([query])[0]
        fetch = limit * 3 if applied and applied.section else limit
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=build_qdrant_filter(applied),
            limit=fetch,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for hit in response.points:
            chunk = chunk_from_payload(hit.payload or {})
            results.append(RetrievedChunk(**chunk.model_dump(), score=float(hit.score)))
        section = applied.section if applied else None
        return apply_section_filter(results, section)[:limit]
