"""Copy existing Qdrant collections to the configured destination (no re-embed)."""

from __future__ import annotations

import argparse
import sys

from qdrant_client.http.models import PointStruct

from app.config import (
    OPENAI_EMBED_DIMENSIONS,
    QDRANT_COLLECTION_RAPTOR,
    QDRANT_COLLECTIONS,
    QDRANT_SOURCE_URL,
    QDRANT_UPSERT_BATCH,
    QDRANT_URL,
)
from app.ingestion.indexer import ensure_collection, get_qdrant_client, upsert_points

_SCROLL = 64


def _collection_names() -> list[str]:
    names = list(QDRANT_COLLECTIONS.values())
    if QDRANT_COLLECTION_RAPTOR not in names:
        names.append(QDRANT_COLLECTION_RAPTOR)
    return names


def _scroll_points(client, collection: str) -> list[PointStruct]:
    points: list[PointStruct] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=_SCROLL,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in batch:
            vector = point.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()))
            points.append(PointStruct(id=point.id, vector=vector, payload=point.payload or {}))
        if offset is None:
            break
    return points


def push_collections(source_url: str, dest_url: str) -> dict[str, int]:
    if source_url.rstrip("/") == dest_url.rstrip("/"):
        raise ValueError("Source and destination Qdrant URLs are the same")
    source = get_qdrant_client(url=source_url, api_key="")
    dest = get_qdrant_client(url=dest_url)
    copied: dict[str, int] = {}
    for name in _collection_names():
        if not source.collection_exists(name):
            print(f"skip {name} (missing on source)", flush=True)
            continue
        info = source.get_collection(name)
        count = int(info.points_count or 0)
        if count == 0:
            print(f"skip {name} (empty)", flush=True)
            continue
        dimensions = OPENAI_EMBED_DIMENSIONS
        vectors = info.config.params.vectors
        if getattr(vectors, "size", None):
            dimensions = int(vectors.size)
        ensure_collection(dest, dimensions, name)
        points = _scroll_points(source, name)
        upsert_points(dest, name, points, batch_size=QDRANT_UPSERT_BATCH)
        copied[name] = len(points)
        print(f"{name}  {len(points)} points → {dest_url}", flush=True)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy local Qdrant collections to QDRANT_URL without re-embedding."
    )
    parser.add_argument("--source", default=QDRANT_SOURCE_URL, help="Source Qdrant URL")
    parser.add_argument("--dest", default=QDRANT_URL, help="Destination Qdrant URL")
    args = parser.parse_args(argv)
    print(f"push {args.source} → {args.dest}", flush=True)
    copied = push_collections(args.source, args.dest)
    if not copied:
        print("No collections copied. Index locally first, or run ingest against QDRANT_URL.", file=sys.stderr)
        return 1
    total = sum(copied.values())
    print(f"done {total} points in {len(copied)} collections", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
