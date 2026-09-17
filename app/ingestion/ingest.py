from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import (
    ACTIVE_CHUNKER,
    CHUNKER_NAMES,
    EMBEDDER_PROVIDER,
    INDEXED_DIR,
    OPENAI_EMBED_MODEL,
    chunked_dir,
    qdrant_collection,
)
from app.ingestion.chunker import ParentChildChunker, get_chunker, write_chunks
from app.ingestion.embed_text import build_embedding_text
from app.ingestion.embedder import get_embedder
from app.ingestion.index import _embed_batches
from app.ingestion.indexer import get_qdrant_client, upsert_chunks, write_index_report
from app.ingestion.loader import load_raw_posts_from_file
from app.ingestion.normalizer import normalize_post, write_normalized_document

_CHUNKER_CHOICES = (*CHUNKER_NAMES, "all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize, chunk, and index a JSON fixture list into Qdrant."
    )
    parser.add_argument("file", help="Path to a JSON list (articles, reviews, or providers)")
    parser.add_argument(
        "content_type",
        choices=("article", "review", "provider"),
        help="Fixture shape to parse",
    )
    parser.add_argument(
        "--chunker",
        choices=_CHUNKER_CHOICES,
        default=ACTIVE_CHUNKER,
        help="Which chunker collection to write, or all",
    )
    args = parser.parse_args(argv)

    posts = load_raw_posts_from_file(args.file, args.content_type)
    if not posts:
        print(f"No records in {args.file}", file=sys.stderr)
        return 1

    normalized = []
    for post in posts:
        document = normalize_post(post)
        path = write_normalized_document(post, document)
        normalized.append((path, document))
        print(f"{path}  ({len(document.sections)} sections)")

    embedder = get_embedder()
    client = get_qdrant_client()
    names = CHUNKER_NAMES if args.chunker == "all" else (args.chunker,)
    for name in names:
        _ingest_chunker(name, normalized, embedder, client, source=str(args.file))
    return 0


def _ingest_chunker(name: str, normalized, embedder, client, source: str) -> int:
    chunker = get_chunker(name)
    directory = chunked_dir(name)
    collection = qdrant_collection(name)
    documents = []
    total = 0
    for path, document in normalized:
        chunks = chunker.chunk(document)
        write_chunks(path.name, chunks, directory=directory)
        if isinstance(chunker, ParentChildChunker):
            write_chunks(path.name, chunker.parents(document), directory=directory / "parents")
        vectors = _embed_batches(embedder, [build_embedding_text(chunk) for chunk in chunks])
        upserted = upsert_chunks(
            chunks,
            vectors,
            embedder.dimensions,
            client=client,
            collection=collection,
        )
        total += upserted
        documents.append(
            {
                "id": document.id,
                "content_type": document.content_type,
                "source": Path(source).name,
                "chunks": upserted,
                "status": "ok",
            }
        )
        print(
            f"{document.content_type} {document.id}  "
            f"{upserted} chunks → {collection} ({name})"
        )
    report_path = write_index_report(
        {
            "collection": collection,
            "chunker": name,
            "source": source,
            "embedder": EMBEDDER_PROVIDER,
            "model": OPENAI_EMBED_MODEL if EMBEDDER_PROVIDER == "openai" else EMBEDDER_PROVIDER,
            "chunk_count": total,
            "documents": documents,
        },
        path=INDEXED_DIR / f"last-run-{name}.json",
    )
    print(f"report {report_path}  ({total} chunks, {name})")
    return total


if __name__ == "__main__":
    sys.exit(main())
