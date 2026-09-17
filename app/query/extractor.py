from __future__ import annotations

import json
from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import OPENAI_API_KEY, QUERY_EXTRACTOR, QUERY_EXTRACTOR_MODEL
from app.models.schemas import ExtractorName, QueryKind
from app.query.catalog import QueryCatalog, get_catalog

_KINDS: tuple[QueryKind, ...] = (
    "factual",
    "comparison",
    "recommendation",
    "pricing",
    "review",
    "multi-hop",
)
_SYSTEM = (
    "Extract retrieval intent from a knowledge-base question. "
    "Use only the allowed providers. Topics must appear in the index headings or body. "
    "Reply with JSON: {\"kind\": \"...\", \"providers\": [], \"topics\": []}."
)


class QueryExtraction(BaseModel):
    kind: QueryKind
    providers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    source: ExtractorName = "openai"


class QueryExtractor(ABC):
    name: ExtractorName

    @abstractmethod
    def extract(self, query: str, catalog: QueryCatalog | None = None) -> QueryExtraction:
        raise NotImplementedError


class HeuristicExtractor(QueryExtractor):
    """Offline fallback only: catalog provider names, no phrase lists, no kind guessing."""

    name = "heuristic"

    def extract(self, query: str, catalog: QueryCatalog | None = None) -> QueryExtraction:
        source = catalog or get_catalog()
        return QueryExtraction(
            kind="factual",
            providers=source.match_providers(query),
            topics=[],
            source="heuristic",
        )


class OpenAIExtractor(QueryExtractor):
    name = "openai"

    def __init__(
        self,
        client=None,
        model: str = QUERY_EXTRACTOR_MODEL,
        api_key: str | None = OPENAI_API_KEY,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._fallback = HeuristicExtractor()

    def extract(self, query: str, catalog: QueryCatalog | None = None) -> QueryExtraction:
        source = catalog or get_catalog()
        try:
            payload = json.loads(self._complete(query, source))
            extracted = QueryExtraction.model_validate({**payload, "source": "openai"})
            return constrain_extraction(extracted, source)
        except Exception:
            return self._fallback.extract(query, source)

    def _complete(self, query: str, catalog: QueryCatalog) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt(query, catalog)},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class BedrockExtractor(QueryExtractor):
    name = "openai"

    def extract(self, query: str, catalog: QueryCatalog | None = None) -> QueryExtraction:
        raise NotImplementedError(
            "Bedrock query extractor is not enabled yet. Use OpenAIExtractor."
        )


def constrain_extraction(extraction: QueryExtraction, catalog: QueryCatalog) -> QueryExtraction:
    allowed_providers = {name.lower(): name for name in catalog.providers}
    providers: list[str] = []
    for name in extraction.providers:
        canonical = allowed_providers.get(name.lower())
        if canonical and canonical not in providers:
            providers.append(canonical)
    topics: list[str] = []
    for name in extraction.topics:
        if catalog.allows_topic(name) and name not in topics:
            topics.append(name)
    kind: QueryKind = extraction.kind if extraction.kind in _KINDS else "factual"
    return extraction.model_copy(update={"kind": kind, "providers": providers, "topics": topics})


def get_extractor(provider: str | None = None) -> QueryExtractor:
    name = (provider or QUERY_EXTRACTOR).lower()
    if name == "heuristic":
        return HeuristicExtractor()
    if name == "openai":
        return OpenAIExtractor()
    if name == "bedrock":
        return BedrockExtractor()
    raise ValueError(f"Unknown query extractor: {name}")


def understand_query(
    query: str,
    *,
    extractor: QueryExtractor | None = None,
    catalog: QueryCatalog | None = None,
) -> QueryExtraction:
    source = catalog or get_catalog()
    backend = extractor or get_extractor()
    return backend.extract(query, source)


def _user_prompt(query: str, catalog: QueryCatalog) -> str:
    headings = ", ".join(catalog.headings[:40]) or "(none)"
    return (
        f"Allowed kinds: {', '.join(_KINDS)}\n"
        f"Allowed providers: {', '.join(catalog.providers) or '(none)'}\n"
        f"Index headings (topics must be grounded in these or the document body): {headings}\n"
        f"Question: {query}"
    )
