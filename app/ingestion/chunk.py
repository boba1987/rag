from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import NORMALIZED_DIR
from app.ingestion.chunker import StructureAwareChunker, count_tokens, write_chunks
from app.models.schemas import NormalizedDocument


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk NormalizedDocument JSON into retrieval units.")
    parser.add_argument("--id", dest="post_id", help="WordPress post id")
    parser.add_argument(
        "--type",
        dest="content_type",
        choices=("article", "review", "provider"),
        help="Document content type",
    )
    parser.add_argument("--all", action="store_true", help="Chunk every file in documents/normalized/")
    args = parser.parse_args(argv)

    if not args.all and not args.post_id:
        parser.error("Provide --id or --all")

    documents = _load_normalized(post_id=args.post_id, content_type=args.content_type)
    if not documents:
        print("No normalized documents matched.", file=sys.stderr)
        return 1

    chunker = StructureAwareChunker()
    for path, document in documents:
        chunks = chunker.chunk(document)
        out = write_chunks(path.name, chunks)
        sizes = [count_tokens(chunk.text) for chunk in chunks]
        print(f"{out}  ({len(chunks)} chunks, tokens={sizes})")
    return 0


def _load_normalized(
    post_id: str | None,
    content_type: str | None,
) -> list[tuple[Path, NormalizedDocument]]:
    if not NORMALIZED_DIR.exists():
        return []
    matches = []
    for path in sorted(NORMALIZED_DIR.glob("*.json")):
        document = NormalizedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if post_id and document.id != str(post_id):
            continue
        if content_type and document.content_type != content_type:
            continue
        matches.append((path, document))
    return matches


if __name__ == "__main__":
    sys.exit(main())
