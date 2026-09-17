from __future__ import annotations

from dataclasses import dataclass

from app.config import DENSE_TOP_K, QDRANT_COLLECTION_RAPTOR
from app.ingestion.embedder import Embedder, get_embedder
from app.ingestion.indexer import get_qdrant_client
from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.retrieval.filters import apply_section_filter, build_qdrant_filter, infer_filters, merge_filters


@dataclass(frozen=True)
class RaptorHit:
    chunk: RetrievedChunk
    node_type: str
    children: tuple[str, ...]
    level: int


def retrieved_from_raptor_payload(payload: dict, score: float) -> RetrievedChunk:
    """Map a RAPTOR Qdrant payload onto RetrievedChunk. Mixed nodes use safe defaults."""
    heading = list(payload.get("heading_path") or [payload.get("section") or "Summary"])
    return RetrievedChunk(
        id=payload["chunk_id"],
        document_id=str(payload.get("document_id") or "mixed"),
        content_type=payload.get("content_type") or "article",
        provider=payload.get("provider"),
        title=payload.get("title") or "",
        section=payload.get("section") or "Summary",
        heading_path=heading or ["Summary"],
        text=payload.get("text") or "",
        source_url=payload.get("source_url"),
        updated_at=payload.get("updated_at"),
        parent_id=payload.get("parent_id"),
        score=score,
    )


def collapse_raptor_hits(hits: list[RaptorHit]) -> list[RetrievedChunk]:
    """Drop a node when an ancestor summary is already in the ranking."""
    by_id = {hit.chunk.id: hit for hit in hits}
    covered: set[str] = set()

    def _cover(node_id: str) -> None:
        hit = by_id.get(node_id)
        if hit is None:
            return
        for child_id in hit.children:
            covered.add(child_id)
            _cover(child_id)

    for hit in hits:
        if hit.node_type == "summary":
            _cover(hit.chunk.id)
    return [hit.chunk for hit in hits if hit.chunk.id not in covered]


class RaptorRetriever:
    """Dense search over the RAPTOR collection, then collapse parent/child overlaps."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        client=None,
        collection: str = QDRANT_COLLECTION_RAPTOR,
        top_k: int = DENSE_TOP_K,
    ) -> None:
        self._embedder = embedder
        self._client = client
        self._collection = collection
        self._top_k = top_k

    def _embedder_or_default(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _client_or_default(self):
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        infer: bool = False,
    ) -> list[RetrievedChunk]:
        limit = top_k or self._top_k
        applied = merge_filters(filters, infer_filters(query)) if infer else filters
        vector = self._embedder_or_default().embed([query])[0]
        fetch = max(limit * 3, limit)
        if applied and applied.section:
            fetch = max(fetch, limit * 3)
        response = self._client_or_default().query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=build_qdrant_filter(applied),
            limit=fetch,
            with_payload=True,
        )
        hits: list[RaptorHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                RaptorHit(
                    chunk=retrieved_from_raptor_payload(payload, float(point.score)),
                    node_type=payload.get("node_type") or "leaf",
                    children=tuple(payload.get("children") or []),
                    level=int(payload.get("level") or 0),
                )
            )
        collapsed = collapse_raptor_hits(hits)
        section = applied.section if applied else None
        return apply_section_filter(collapsed, section)[:limit]
