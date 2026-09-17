from __future__ import annotations

import json
from pathlib import Path

from app.config import chunked_dir
from app.models.schemas import Chunk, RetrievalFilters, RetrievedChunk


def parents_dir(chunker: str = "parent_child") -> Path:
    return chunked_dir(chunker) / "parents"


class FileParentStore:
    """Parents written by the chunk/ingest CLIs. Children are what Qdrant/BM25 search."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or parents_dir()
        self._index: dict[str, Chunk] | None = None

    def get(self, parent_id: str) -> Chunk | None:
        return self._loaded().get(parent_id)

    def _loaded(self) -> dict[str, Chunk]:
        if self._index is None:
            self._index = _index_parents(self._directory)
        return self._index


def expand_to_parents(
    chunks: list[RetrievedChunk],
    store: FileParentStore | None = None,
) -> list[RetrievedChunk]:
    """Replace child hits with their parent section. Sibling hits collapse to one parent."""
    parents = store or FileParentStore()
    expanded: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if not chunk.parent_id:
            expanded.append(chunk)
            continue
        parent = parents.get(chunk.parent_id)
        if parent is None:
            expanded.append(chunk)
            continue
        if parent.id in seen:
            continue
        seen.add(parent.id)
        expanded.append(RetrievedChunk(**parent.model_dump(), score=chunk.score))
    return expanded


class ParentExpandingRetriever:
    """Search children, then supply parent text to the generator."""

    def __init__(self, inner, store: FileParentStore | None = None) -> None:
        self._inner = inner
        self._store = store or FileParentStore()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        infer: bool = False,
    ) -> list[RetrievedChunk]:
        hits = self._inner.search(query, top_k=top_k, filters=filters, infer=infer)
        return expand_to_parents(hits, self._store)


def _index_parents(directory: Path) -> dict[str, Chunk]:
    index: dict[str, Chunk] = {}
    if not directory.is_dir():
        return index
    for path in sorted(directory.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            parent = Chunk.model_validate(row)
            index[parent.id] = parent
    return index
