from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import ACTIVE_CHUNKER, CHUNKER_NAMES, NORMALIZED_DIR, chunked_dir
from app.ingestion.chunker import ParentChildChunker, count_tokens, get_chunker, write_chunks
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
    parser.add_argument(
        "--chunker",
        choices=CHUNKER_NAMES,
        default=ACTIVE_CHUNKER,
        help="Which chunker and output directory to use",
    )
    args = parser.parse_args(argv)

    if not args.all and not args.post_id:
        parser.error("Provide --id or --all")

    documents = _load_normalized(post_id=args.post_id, content_type=args.content_type)
    if not documents:
        print("No normalized documents matched.", file=sys.stderr)
        return 1

    chunker = get_chunker(args.chunker)
    directory = chunked_dir(args.chunker)
    for path, document in documents:
        chunks = chunker.chunk(document)
        out = write_chunks(path.name, chunks, directory=directory)
        sizes = [count_tokens(chunk.text) for chunk in chunks]
        print(f"{out}  ({len(chunks)} chunks, tokens={sizes})")
        if isinstance(chunker, ParentChildChunker):
            parents = chunker.parents(document)
            parent_out = write_chunks(path.name, parents, directory=directory / "parents")
            print(f"{parent_out}  ({len(parents)} parents)")
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
