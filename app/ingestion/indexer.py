from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import (
    INDEXED_DIR,
    OPENAI_EMBED_DIMENSIONS,
    QDRANT_COLLECTION,
    QDRANT_COLLECTIONS,
    QDRANT_URL,
)
from app.models.schemas import Chunk


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"getvoip-rag:{chunk_id}"))


def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url, check_compatibility=False)


def ensure_chunker_collections(
    client: QdrantClient | None = None,
    dimensions: int = OPENAI_EMBED_DIMENSIONS,
) -> dict[str, str]:
    """Create one Qdrant collection per chunker. Existing collections are left as-is."""
    qdrant = client or get_qdrant_client()
    for collection in QDRANT_COLLECTIONS.values():
        ensure_collection(qdrant, dimensions, collection)
    return dict(QDRANT_COLLECTIONS)


def ensure_collection(
    client: QdrantClient,
    dimensions: int,
    collection: str = QDRANT_COLLECTION,
) -> None:
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
    )


def chunk_from_payload(payload: dict) -> Chunk:
    return Chunk(
        id=payload["chunk_id"],
        document_id=payload["document_id"],
        content_type=payload["content_type"],
        provider=payload.get("provider"),
        title=payload["title"],
        section=payload["section"],
        heading_path=list(payload.get("heading_path") or [payload["section"]]),
        text=payload["text"],
        source_url=payload.get("source_url"),
        updated_at=payload.get("updated_at"),
        parent_id=payload.get("parent_id"),
    )


def chunk_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "content_type": chunk.content_type,
        "provider": chunk.provider,
        "title": chunk.title,
        "section": chunk.section,
        "heading_path": chunk.heading_path,
        "text": chunk.text,
        "source_url": chunk.source_url,
        "updated_at": chunk.updated_at,
        "parent_id": chunk.parent_id,
    }


def upsert_chunks(
    chunks: list[Chunk],
    vectors: list[list[float]],
    dimensions: int,
    client: QdrantClient | None = None,
    collection: str = QDRANT_COLLECTION,
) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")
    qdrant = client or get_qdrant_client()
    ensure_collection(qdrant, dimensions, collection)
    points = [
        PointStruct(
            id=chunk_point_id(chunk.id),
            vector=vector,
            payload=chunk_payload(chunk),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    if points:
        qdrant.upsert(collection_name=collection, points=points)
    return len(points)


def write_index_report(report: dict, path: Path | None = None) -> Path:
    INDEXED_DIR.mkdir(parents=True, exist_ok=True)
    target = path or INDEXED_DIR / "last-run.json"
    payload = {"ran_at": datetime.now(timezone.utc).isoformat(), **report}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
