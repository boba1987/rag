from __future__ import annotations

from typing import Literal

QueryKind = Literal[
    "factual",
    "comparison",
    "recommendation",
    "pricing",
    "review",
    "multi-hop",
]

_PROVIDERS = ("ringcentral", "ring central", "nextiva", "five9", "dialpad")
_COMPARE = (" vs ", "versus", "compare", "compared", "difference between", "better than")
_RECOMMEND = ("should i", "recommend", "right for", "good fit", "who is", "which provider")
_PRICING = ("how much", "pricing", "price", "cost", "per user", "per month")
_REVIEW = ("review", "customers say", "reviewers", "what do people")
_SECOND_TOPIC = ("integrat", "salesforce", "support", "feature")


def classify_query(query: str) -> QueryKind:
    """Heuristic query type. Conservative; no LLM."""
    text = f" {query.lower()} "
    providers = _provider_count(text)
    comparison = any(hint in text for hint in _COMPARE)
    pricing = any(hint in text for hint in _PRICING)
    other_topic = any(hint in text for hint in _SECOND_TOPIC)
    if providers >= 2 and (pricing and other_topic):
        return "multi-hop"
    if comparison:
        return "comparison"
    if any(hint in text for hint in _RECOMMEND):
        return "recommendation"
    if pricing:
        return "pricing"
    if any(hint in text for hint in _REVIEW):
        return "review"
    return "factual"


def _provider_count(text: str) -> int:
    found = set()
    for name in _PROVIDERS:
        if name in text:
            found.add(name.replace(" ", ""))
    return len(found)
