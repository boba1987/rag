from __future__ import annotations

import re

from app.query.catalog import QueryCatalog, get_catalog
from app.query.extractor import QueryExtraction

_FILLER = re.compile(
    r"^(please\s+)?(can you|could you|would you)?\s*"
    r"(tell me|explain|i (want|need|would like) to know)\s+",
    re.IGNORECASE,
)


def rewrite_query(
    query: str,
    extraction: QueryExtraction | None = None,
    catalog: QueryCatalog | None = None,
) -> str:
    """Turn a chatty or messy question into a tighter retrieval query."""
    source = catalog or get_catalog()
    text = _FILLER.sub("", query.strip())
    text = re.sub(r"\s+", " ", text).strip(" ?")
    if not text:
        return query.strip()
    text = _canonicalize_providers(text, source)
    if extraction is None:
        return text
    if extraction.kind == "pricing":
        text = _ensure_terms(text, "pricing", "cost")
    elif extraction.kind == "review":
        text = _ensure_terms(text, "reviews")
    elif extraction.kind == "comparison":
        text = _ensure_terms(text, "compare")
    return text


def _canonicalize_providers(text: str, catalog: QueryCatalog) -> str:
    rewritten = text
    for needle, name in catalog.aliases:
        rewritten = re.sub(rf"\b{re.escape(needle)}\b", name, rewritten, flags=re.IGNORECASE)
    return rewritten


def _ensure_terms(text: str, *terms: str) -> str:
    lower = text.lower()
    missing = [term for term in terms if term not in lower]
    if not missing:
        return text
    return f"{text} {' '.join(missing)}".strip()
