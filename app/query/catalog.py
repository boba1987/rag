from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import NORMALIZED_DIR, QDRANT_COLLECTION
from app.models.schemas import NormalizedDocument

_CAMEL = re.compile(r"([a-z])([A-Z])")


@dataclass(frozen=True)
class QueryCatalog:
    """Providers and headings from Qdrant (or a normalized fixture). No static cue file."""

    providers: tuple[str, ...]
    aliases: tuple[tuple[str, str], ...]
    headings: tuple[str, ...] = ()
    corpus: str = ""

    def match_providers(self, query: str) -> list[str]:
        text = query.lower()
        found: list[str] = []
        for needle, name in self.aliases:
            if re.search(rf"\b{re.escape(needle)}\b", text) and name not in found:
                found.append(name)
        return found

    def allows_topic(self, topic: str) -> bool:
        blob = self.corpus or " ".join(self.headings).lower()
        if not blob:
            return True
        needle = topic.lower().strip()
        if not needle:
            return False
        if needle in blob:
            return True
        words = [word for word in re.findall(r"[a-z0-9]+", needle) if len(word) > 3]
        return bool(words) and all(word in blob for word in words)


def aliases_for(name: str) -> list[str]:
    spaced = _CAMEL.sub(r"\1 \2", name)
    return sorted(
        {name.lower(), name.lower().replace(" ", ""), spaced.lower()},
        key=len,
        reverse=True,
    )


def load_catalog(
    normalized_dir: Path | None = None,
    client=None,
    collection: str | None = None,
) -> QueryCatalog:
    if normalized_dir is not None:
        return catalog_from_normalized(normalized_dir)
    try:
        catalog = catalog_from_qdrant(client=client, collection=collection)
        if catalog.providers or catalog.headings:
            return catalog
    except Exception:
        pass
    if NORMALIZED_DIR.is_dir() and any(NORMALIZED_DIR.glob("*.json")):
        return catalog_from_normalized(NORMALIZED_DIR)
    return QueryCatalog(providers=(), aliases=())


def catalog_from_qdrant(client=None, collection: str | None = None) -> QueryCatalog:
    """Read distinct providers, sections, and text from the active Qdrant collection."""
    from app.ingestion.indexer import get_qdrant_client

    qdrant = client or get_qdrant_client()
    name = collection or QDRANT_COLLECTION
    if not qdrant.collection_exists(name):
        return QueryCatalog(providers=(), aliases=())
    providers: set[str] = set()
    headings: list[str] = []
    parts: list[str] = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=name,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("provider"):
                providers.add(str(payload["provider"]))
            title = str(payload.get("title") or "")
            section = str(payload.get("section") or "")
            if section:
                headings.append(section)
            text = str(payload.get("text") or "")
            parts.extend([title, section, text])
        if offset is None:
            break
    return _build_catalog(
        providers,
        headings=tuple(dict.fromkeys(headings)),
        corpus="\n".join(parts).lower(),
    )


def catalog_from_normalized(normalized_dir: Path) -> QueryCatalog:
    providers: set[str] = set()
    headings: list[str] = []
    parts: list[str] = []
    for path in sorted(normalized_dir.glob("*.json")):
        document = NormalizedDocument.model_validate_json(path.read_text())
        if document.provider:
            providers.add(document.provider)
        parts.append(document.title)
        for section in document.sections:
            if section.heading:
                headings.append(section.heading)
            parts.append(section.heading)
            parts.append(section.text)
    return _build_catalog(providers, headings=tuple(dict.fromkeys(headings)), corpus="\n".join(parts).lower())


def _build_catalog(
    providers: set[str],
    headings: tuple[str, ...] = (),
    corpus: str = "",
) -> QueryCatalog:
    aliases: list[tuple[str, str]] = []
    for name in sorted(providers):
        for needle in aliases_for(name):
            aliases.append((needle, name))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return QueryCatalog(
        providers=tuple(sorted(providers)),
        aliases=tuple(aliases),
        headings=headings,
        corpus=corpus,
    )


@lru_cache(maxsize=1)
def get_catalog() -> QueryCatalog:
    return load_catalog()


def reset_catalog() -> None:
    get_catalog.cache_clear()
