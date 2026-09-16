from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import CHUNKED_DIR, EMBEDDER_PROVIDER, OPENAI_EMBED_MODEL, QDRANT_COLLECTION
from app.ingestion.embed_text import build_embedding_text
from app.ingestion.embedder import get_embedder
from app.ingestion.indexer import get_qdrant_client, upsert_chunks, write_index_report
from app.models.schemas import Chunk

_EMBED_BATCH = 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed chunk JSON and upsert into Qdrant.")
    parser.add_argument("--id", dest="post_id", help="WordPress post id")
    parser.add_argument(
        "--type",
        dest="content_type",
        choices=("article", "review", "provider"),
        help="Document content type",
    )
    parser.add_argument("--all", action="store_true", help="Index every file in documents/chunked/")
    args = parser.parse_args(argv)

    if not args.all and not args.post_id:
        parser.error("Provide --id or --all")

    files = _load_chunk_files(post_id=args.post_id, content_type=args.content_type)
    if not files:
        print("No chunk files matched. Run python -m app.ingestion.chunk first.", file=sys.stderr)
        return 1

    embedder = get_embedder()
    client = get_qdrant_client()
    documents = []
    total = 0

    for path, chunks in files:
        vectors = _embed_batches(embedder, [build_embedding_text(chunk) for chunk in chunks])
        upserted = upsert_chunks(
            chunks,
            vectors,
            embedder.dimensions,
            client=client,
            collection=QDRANT_COLLECTION,
        )
        total += upserted
        doc = {
            "id": chunks[0].document_id,
            "content_type": chunks[0].content_type,
            "source": path.name,
            "chunks": upserted,
            "status": "ok",
        }
        documents.append(doc)
        print(f"{path.name}  ({upserted} chunks → {QDRANT_COLLECTION})")

    report_path = write_index_report(
        {
            "collection": QDRANT_COLLECTION,
            "embedder": EMBEDDER_PROVIDER,
            "model": OPENAI_EMBED_MODEL if EMBEDDER_PROVIDER == "openai" else EMBEDDER_PROVIDER,
            "chunk_count": total,
            "documents": documents,
        }
    )
    print(f"report {report_path}  ({total} chunks total)")
    return 0


def _embed_batches(embedder, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        vectors.extend(embedder.embed(texts[start : start + _EMBED_BATCH]))
    return vectors


def _load_chunk_files(
    post_id: str | None,
    content_type: str | None,
) -> list[tuple[Path, list[Chunk]]]:
    if not CHUNKED_DIR.exists():
        return []
    matches = []
    for path in sorted(CHUNKED_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(row) for row in raw]
        if not chunks:
            continue
        if post_id and chunks[0].document_id != str(post_id):
            continue
        if content_type and chunks[0].content_type != content_type:
            continue
        matches.append((path, chunks))
    return matches


if __name__ == "__main__":
    sys.exit(main())
