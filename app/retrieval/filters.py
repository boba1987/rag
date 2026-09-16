from __future__ import annotations

import re

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.models.schemas import RetrievalFilters, RetrievedChunk

_PROVIDERS = (("ring central", "RingCentral"), ("ringcentral", "RingCentral"), ("nextiva", "Nextiva"))
_PRICING_HINTS = ("how much", "pricing", "price", "cost", "per user", "per month", "$")
_INTEGRATION_HINTS = (
    "integrat",
    "salesforce",
    "zendesk",
    "hubspot",
    "microsoft 365",
    "google workspace",
    "teams",
)
_SUPPORT_HINTS = ("support", "customer service")
_REVIEW_HINTS = ("review", "customers say", "reviewers")


def infer_filters(query: str) -> RetrievalFilters:
    """Heuristic filters from the question. Conservative: only Nextiva/RingCentral providers."""
    text = query.lower()
    provider = _infer_provider(text)
    section = _infer_section(text)
    content_type = _infer_content_type(text, provider=provider, section=section)
    return RetrievalFilters(provider=provider, section=section, content_type=content_type)


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


def _infer_provider(text: str) -> str | None:
    for needle, name in _PROVIDERS:
        if needle in text:
            return name
    return None


def _infer_section(text: str) -> str | None:
    if any(hint in text for hint in _PRICING_HINTS):
        return "Pricing"
    if any(hint in text for hint in _INTEGRATION_HINTS):
        return "Integration"
    if any(hint in text for hint in _SUPPORT_HINTS):
        return "Support"
    return None


def _infer_content_type(text: str, provider: str | None, section: str | None) -> str | None:
    if any(hint in text for hint in _REVIEW_HINTS):
        return "review"
    if provider and section in {"Pricing", "Integration"}:
        return "provider"
    if re.search(r"\b(article|buying guide)\b", text):
        return "article"
    return None
