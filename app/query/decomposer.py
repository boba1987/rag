from __future__ import annotations

from app.query.classifier import classify_query, mentioned_providers
from app.query.rewriter import rewrite_query

_PRICING = ("how much", "pricing", "price", "cost", "per user", "per month")


def mentioned_topics(query: str) -> list[str]:
    text = query.lower()
    topics: list[str] = []
    if any(hint in text for hint in _PRICING):
        topics.append("pricing")
    if "salesforce" in text:
        topics.append("Salesforce integration")
    elif "integrat" in text:
        topics.append("integrations")
    if "support" in text:
        topics.append("support")
    return topics


def decompose_query(query: str) -> list[str]:
    """Split a multi-provider / multi-topic question into retrieval sub-queries."""
    rewritten = rewrite_query(query)
    providers = mentioned_providers(rewritten)
    topics = mentioned_topics(rewritten)
    if len(providers) >= 2 and topics:
        return [f"{provider} {topic}" for topic in topics for provider in providers]
    return [rewritten]


def expand_queries(query: str) -> list[str]:
    """Queries to retrieve. Uses decomposition when it splits; otherwise rewrite plus one alternate."""
    parts = decompose_query(query)
    if len(parts) > 1:
        return parts
    rewritten = parts[0]
    extras: list[str] = []
    kind = classify_query(query)
    if kind == "pricing":
        extras.append(f"{rewritten} plans per user")
    elif kind == "review":
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
