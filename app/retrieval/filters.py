from __future__ import annotations

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.models.schemas import ContentType, RetrievalFilters, RetrievedChunk
from app.query.catalog import get_catalog
from app.query.extractor import QueryExtraction

_KIND_CONTENT_TYPE: dict[str, ContentType] = {
    "review": "review",
}


def infer_filters(query: str, extraction: QueryExtraction | None = None) -> RetrievalFilters:
    """Filters from catalog providers and extraction kind. No cue or synonym lists."""
    if extraction is not None:
        providers = extraction.providers
        provider = providers[0] if len(providers) == 1 else None
        content_type = _KIND_CONTENT_TYPE.get(extraction.kind)
        if content_type is None and provider:
            content_type = "provider"
        return RetrievalFilters(provider=provider, content_type=content_type)
    providers = get_catalog().match_providers(query)
    provider = providers[0] if len(providers) == 1 else None
    return RetrievalFilters(provider=provider)


def merge_filters(explicit: RetrievalFilters | None, inferred: RetrievalFilters | None) -> RetrievalFilters | None:
    """Explicit values win. Returns None when nothing is set."""
    if explicit is None and inferred is None:
        return None
    data = (inferred.model_dump() if inferred else {}) | {
        key: value for key, value in (explicit.model_dump() if explicit else {}).items() if value is not None
    }
    merged = RetrievalFilters.model_validate(data)
    if not any((merged.provider, merged.section, merged.content_type, merged.document_id)):
        return None
    return merged


def build_qdrant_filter(filters: RetrievalFilters | None) -> Filter | None:
    """AND exact payload matches. Section is applied later as a substring hint."""
    if filters is None:
        return None
    must: list[FieldCondition] = []
    if filters.provider:
        must.append(FieldCondition(key="provider", match=MatchValue(value=filters.provider)))
    if filters.content_type:
        must.append(FieldCondition(key="content_type", match=MatchValue(value=filters.content_type)))
    if filters.document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=filters.document_id)))
    if not must:
        return None
    return Filter(must=must)


def matches_section(chunk: RetrievedChunk, section: str | None) -> bool:
    if not section:
        return True
    needle = section.lower()
    if needle in chunk.section.lower():
        return True
    return any(needle in part.lower() for part in chunk.heading_path)


def apply_section_filter(chunks: list[RetrievedChunk], section: str | None) -> list[RetrievedChunk]:
    """Keep chunks whose heading contains the hint. If none match, keep the original ranking."""
    if not section:
        return chunks
    matched = [chunk for chunk in chunks if matches_section(chunk, section)]
    return matched or chunks
