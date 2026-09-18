from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from openai import OpenAI

from app.config import OPENAI_API_KEY, QUERY_REWRITER, QUERY_REWRITER_MODEL
from app.models.schemas import ExtractorName
from app.query.catalog import QueryCatalog, get_catalog
from app.query.extractor import QueryExtraction

_FILLER = re.compile(
    r"^(please\s+)?(can you|could you|would you)?\s*"
    r"(tell me|explain|i (want|need|would like) to know)\s+",
    re.IGNORECASE,
)
_SYSTEM = (
    "Rewrite the question as a single short retrieval query for a VoIP/UCaaS knowledge base. "
    "Keep every provider named in the question. "
    "Do not add providers that are not named in the question. "
    "Do not substitute a different vendor. "
    "Do not invent prices, features, or facts. "
    "Reply with JSON: {\"query\": \"...\"}."
)


class QueryRewriter(ABC):
    name: ExtractorName

    @abstractmethod
    def rewrite(
        self,
        query: str,
        extraction: QueryExtraction | None = None,
        catalog: QueryCatalog | None = None,
    ) -> str:
        raise NotImplementedError


class HeuristicRewriter(QueryRewriter):
    """Offline fallback: strip filler, canonicalize catalog aliases, add kind terms."""

    name = "heuristic"

    def rewrite(
        self,
        query: str,
        extraction: QueryExtraction | None = None,
        catalog: QueryCatalog | None = None,
    ) -> str:
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


class OpenAIRewriter(QueryRewriter):
    name = "openai"

    def __init__(
        self,
        client=None,
        model: str = QUERY_REWRITER_MODEL,
        api_key: str | None = OPENAI_API_KEY,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._fallback = HeuristicRewriter()

    def rewrite(
        self,
        query: str,
        extraction: QueryExtraction | None = None,
        catalog: QueryCatalog | None = None,
    ) -> str:
        source = catalog or get_catalog()
        if not query.strip():
            return self._fallback.rewrite(query, extraction, source)
        try:
            payload = json.loads(self._complete(query, extraction, source))
            rewritten = str(payload.get("query") or payload.get("rewritten") or "").strip()
            return constrain_rewrite(query, rewritten, source)
        except Exception:
            return self._fallback.rewrite(query, extraction, source)

    def _complete(
        self,
        query: str,
        extraction: QueryExtraction | None,
        catalog: QueryCatalog,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt(query, extraction, catalog)},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class BedrockRewriter(QueryRewriter):
    name = "openai"

    def rewrite(
        self,
        query: str,
        extraction: QueryExtraction | None = None,
        catalog: QueryCatalog | None = None,
    ) -> str:
        raise NotImplementedError("Bedrock query rewriter is not enabled yet. Use OpenAIRewriter.")


def constrain_rewrite(query: str, rewritten: str, catalog: QueryCatalog) -> str:
    text = re.sub(r"\s+", " ", (rewritten or "").strip()).strip(" ?")
    if not text:
        raise ValueError("empty rewrite")
    mentioned = catalog.match_providers(query)
    produced = catalog.match_providers(text)
    extra = [name for name in produced if name not in mentioned]
    if extra:
        raise ValueError(f"invented providers: {extra}")
    text = _canonicalize_providers(text, catalog)
    missing = [name for name in mentioned if name not in catalog.match_providers(text)]
    if missing:
        text = f"{text} {' '.join(missing)}".strip()
    return text


def get_rewriter(provider: str | None = None) -> QueryRewriter:
    name = (provider or QUERY_REWRITER).lower()
    if name == "heuristic":
        return HeuristicRewriter()
    if name == "openai":
        try:
            return OpenAIRewriter()
        except Exception:
            return HeuristicRewriter()
    if name == "bedrock":
        return BedrockRewriter()
    raise ValueError(f"Unknown query rewriter: {name}")


def rewrite_query(
    query: str,
    extraction: QueryExtraction | None = None,
    catalog: QueryCatalog | None = None,
    rewriter: QueryRewriter | None = None,
) -> str:
    """Turn a chatty or messy question into a tighter retrieval query."""
    source = catalog or get_catalog()
    backend = rewriter or get_rewriter()
    return backend.rewrite(query, extraction, source)


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


def _user_prompt(
    query: str,
    extraction: QueryExtraction | None,
    catalog: QueryCatalog,
) -> str:
    mentioned = catalog.match_providers(query)
    kind = extraction.kind if extraction else "unknown"
    topics = ", ".join(extraction.topics) if extraction and extraction.topics else "(none)"
    providers = ", ".join(mentioned) or "(none)"
    return (
        f"Kind: {kind}\n"
        f"Providers named in the question (use only these; do not add others): {providers}\n"
        f"Topics: {topics}\n"
        f"Question: {query}"
    )
