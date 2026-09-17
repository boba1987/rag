from __future__ import annotations

from app.models.schemas import QueryKind
from app.query.catalog import QueryCatalog, get_catalog
from app.query.extractor import understand_query


def classify_query(query: str, catalog: QueryCatalog | None = None, extractor=None) -> QueryKind:
    """Query type from the cheap LLM extractor, constrained to the index catalog."""
    return understand_query(query, extractor=extractor, catalog=catalog).kind


def mentioned_providers(query: str, catalog: QueryCatalog | None = None) -> list[str]:
    return (catalog or get_catalog()).match_providers(query)
