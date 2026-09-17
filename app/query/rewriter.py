from __future__ import annotations

import re

from app.query.classifier import classify_query

_FILLER = re.compile(
    r"^(please\s+)?(can you|could you|would you)?\s*"
    r"(tell me|explain|i (want|need|would like) to know)\s+",
    re.IGNORECASE,
)
_PROVIDER_CANON = (
    (re.compile(r"\bring\s+central\b", re.IGNORECASE), "RingCentral"),
    (re.compile(r"\bringcentral\b", re.IGNORECASE), "RingCentral"),
    (re.compile(r"\bnextiva\b", re.IGNORECASE), "Nextiva"),
    (re.compile(r"\bfive9\b", re.IGNORECASE), "Five9"),
    (re.compile(r"\bdialpad\b", re.IGNORECASE), "Dialpad"),
)


def rewrite_query(query: str) -> str:
    """Turn a chatty or messy question into a tighter retrieval query. No LLM."""
    text = _FILLER.sub("", query.strip())
    text = re.sub(r"\s+", " ", text).strip(" ?")
    if not text:
        return query.strip()
    text = _canonicalize_providers(text)
    kind = classify_query(text)
    if kind == "pricing":
        text = _ensure_terms(text, "pricing", "cost")
    elif kind == "review":
        text = _ensure_terms(text, "reviews")
    elif kind == "comparison":
        text = _ensure_terms(text, "compare")
    return text


def _canonicalize_providers(text: str) -> str:
    rewritten = text
    for pattern, name in _PROVIDER_CANON:
        rewritten = pattern.sub(name, rewritten)
    return rewritten


def _ensure_terms(text: str, *terms: str) -> str:
    lower = text.lower()
    missing = [term for term in terms if term not in lower]
    if not missing:
        return text
    return f"{text} {' '.join(missing)}".strip()
