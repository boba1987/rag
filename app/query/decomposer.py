from __future__ import annotations

from app.query.catalog import QueryCatalog, get_catalog
from app.query.extractor import QueryExtraction, understand_query
from app.query.rewriter import rewrite_query


def decompose_query(
    query: str,
    extraction: QueryExtraction | None = None,
    catalog: QueryCatalog | None = None,
    extractor=None,
) -> list[str]:
    """Split a multi-provider / multi-topic question into retrieval sub-queries."""
    source = catalog or get_catalog()
    extracted = extraction or understand_query(query, extractor=extractor, catalog=source)
    rewritten = rewrite_query(query, extraction=extracted, catalog=source)
    providers = extracted.providers or source.match_providers(rewritten)
    topics = extracted.topics
    if len(providers) >= 2 and topics:
        return [f"{provider} {topic}" for topic in topics for provider in providers]
    return [rewritten]


def expand_queries(
    query: str,
    extraction: QueryExtraction | None = None,
    catalog: QueryCatalog | None = None,
    extractor=None,
) -> list[str]:
    """Queries to retrieve. Uses decomposition when it splits; otherwise rewrite plus one alternate."""
    source = catalog or get_catalog()
    extracted = extraction or understand_query(query, extractor=extractor, catalog=source)
    parts = decompose_query(query, extraction=extracted, catalog=source)
    if len(parts) > 1:
        return parts
    rewritten = parts[0]
    extras: list[str] = []
    if extracted.kind == "pricing":
        extras.append(f"{rewritten} plans per user")
    elif extracted.kind == "review":
        extras.append(f"{rewritten} customer support")
    return _unique([rewritten, *extras])


def _unique(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(query)
    return ordered
