from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from openai import OpenAI

from app.config import (
    BGE_RERANKER_MODEL,
    CROSS_ENCODER_MODEL,
    OPENAI_API_KEY,
    RERANK_CANDIDATES,
    RERANK_PASSAGE_CHARS,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
)
from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.observability.tracing import span

_TOKEN_RE = re.compile(r"[a-z0-9$]+")
_OPENAI_SYSTEM = (
    "Rank passages for answering the question. "
    "Reply with JSON: {\"order\": [best_index, ...]} using each passage index exactly once. "
    "Do not invent passages."
)


class Reranker(ABC):
    """Reorder retrieved chunks. Cross-encoder and BGE now; ColBERT-style later."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Score (query, chunk) pairs. Inject predict() in tests; load a model only when needed."""

    def __init__(
        self,
        predict=None,
        model: str = CROSS_ENCODER_MODEL,
        top_k: int = RERANK_TOP_K,
        fallback: Reranker | None = None,
    ) -> None:
        self._predict = predict
        self._model_name = model
        self._top_k = top_k
        self._fallback = fallback

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        limit = top_k or self._top_k
        if self._predict is None:
            return self._resolve_fallback().rerank(query, chunks, top_k=limit)
        scores = self._score_pairs([(query, chunk.text) for chunk in chunks])
        ranked = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return ranked[:limit]

    def _resolve_fallback(self) -> Reranker:
        if self._fallback is not None:
            return self._fallback
        if OPENAI_API_KEY:
            return OpenAIReranker(top_k=self._top_k)
        return LexicalReranker(top_k=self._top_k)

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._predict is None:
            raise RuntimeError("CrossEncoderReranker has no injected predict()")
        return [float(score) for score in self._predict(pairs)]


class LexicalReranker(Reranker):
    """Token-overlap fallback when sentence-transformers / torch is unavailable."""

    def __init__(self, top_k: int = RERANK_TOP_K) -> None:
        self._top_k = top_k

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        limit = top_k or self._top_k
        ranked = [
            chunk.model_copy(update={"score": _lexical_score(query, chunk.text)})
            for chunk in chunks
        ]
        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return ranked[:limit]


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(_TOKEN_RE.findall(query.lower()))
    if not query_tokens:
        return 0.0
    passage = set(_TOKEN_RE.findall(text.lower()))
    return len(query_tokens & passage) / len(query_tokens)


class OpenAIReranker(Reranker):
    """Listwise LLM rerank. This is the live reranker; MiniLM is not used."""

    def __init__(
        self,
        client=None,
        model: str = RERANKER_MODEL,
        api_key: str | None = OPENAI_API_KEY,
        top_k: int = RERANK_TOP_K,
        passage_chars: int = RERANK_PASSAGE_CHARS,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._top_k = top_k
        self._passage_chars = passage_chars
        self._fallback = LexicalReranker(top_k=top_k)

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        limit = top_k or self._top_k
        with span(
            "rerank.llm",
            as_type="generation",
            model=self._model,
            input={"query": query, "passages": len(chunks)},
            metadata={"passage_chars": self._passage_chars, "top_k": limit},
        ) as llm_span:
            try:
                order = self._rank(query, chunks, llm_span)
            except Exception as error:
                llm_span.update(
                    output={"fallback": "lexical", "error": str(error)},
                    metadata={"fallback": True},
                )
                return self._fallback.rerank(query, chunks, top_k=limit)
            llm_span.update(output={"order": order})
        ranked = _apply_order(chunks, order)
        total = max(len(ranked), 1)
        return [
            chunk.model_copy(update={"score": float(total - index)})
            for index, chunk in enumerate(ranked[:limit])
        ]

    def _rank(self, query: str, chunks: list[RetrievedChunk], observation=None) -> list[int]:
        payload = json.loads(self._complete(query, chunks, observation))
        raw = payload.get("order") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            raise ValueError("rerank response missing order")
        order: list[int] = []
        for item in raw:
            index = int(item)
            if 0 <= index < len(chunks) and index not in order:
                order.append(index)
        if not order:
            raise ValueError("rerank response had no valid indices")
        return order

    def _complete(self, query: str, chunks: list[RetrievedChunk], observation=None) -> str:
        passages = "\n".join(
            f"{index}. {chunk.title} / {chunk.section}: {chunk.text[:self._passage_chars]}"
            for index, chunk in enumerate(chunks)
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _OPENAI_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nPassages:\n{passages}",
                },
            ],
        )
        if observation is not None:
            _record_usage(observation, getattr(response, "usage", None))
        return (response.choices[0].message.content or "").strip()


def _record_usage(observation, usage) -> None:
    """Report OpenAI token counts so the rerank call shows real cost in Langfuse."""
    if usage is None:
        return
    details = {
        "input": getattr(usage, "prompt_tokens", None),
        "output": getattr(usage, "completion_tokens", None),
        "total": getattr(usage, "total_tokens", None),
    }
    details = {key: value for key, value in details.items() if value is not None}
    if details:
        observation.update(usage_details=details, metadata={"usage": details})


def _apply_order(chunks: list[RetrievedChunk], order: list[int]) -> list[RetrievedChunk]:
    ranked = [chunks[index] for index in order]
    seen = set(order)
    ranked.extend(chunk for index, chunk in enumerate(chunks) if index not in seen)
    return ranked


class BGEReranker(CrossEncoderReranker):
    """BGE pair scorer. Same rerank() contract; default model is BAAI/bge-reranker-base."""

    def __init__(
        self,
        predict=None,
        model: str = BGE_RERANKER_MODEL,
        top_k: int = RERANK_TOP_K,
    ) -> None:
        super().__init__(predict=predict, model=model, top_k=top_k)


def get_reranker(provider: str | None = None) -> Reranker:
    name = (provider or RERANKER_PROVIDER).lower()
    if name in {"lexical", "heuristic"}:
        return LexicalReranker()
    if name in {"openai", "gpt", "cross-encoder", "cross_encoder", "bge"}:
        return OpenAIReranker()
    raise ValueError(f"Unknown reranker provider: {name}")


class RerankRetriever:
    """Hybrid retrieve a large pool, then rerank to top_k. Does not live in LangGraph."""

    def __init__(
        self,
        retriever=None,
        reranker: Reranker | None = None,
        candidates: int = RERANK_CANDIDATES,
        top_k: int = RERANK_TOP_K,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker if reranker is not None else get_reranker()
        self._candidates = candidates
        self._top_k = top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        infer: bool = False,
    ) -> list[RetrievedChunk]:
        limit = top_k or self._top_k
        searcher = self._retriever
        if searcher is None:
            from app.retrieval.hybrid import HybridRetriever

            searcher = HybridRetriever()
            self._retriever = searcher
        pool = searcher.search(
            query, top_k=self._candidates, filters=filters, infer=infer
        )
        with span(
            "rerank",
            input={"query": query, "pool_ids": [chunk.id for chunk in pool]},
            metadata={
                "reranker": type(self._reranker).__name__,
                "candidates": self._candidates,
                "top_k": limit,
            },
            model=getattr(self._reranker, "model", None),
        ) as rerank_span:
            ranked = self._reranker.rerank(query, pool, top_k=limit)
            rerank_span.update(
                output={
                    "chunk_ids": [chunk.id for chunk in ranked],
                    "document_ids": [chunk.document_id for chunk in ranked],
                    "kept": len(ranked),
                    "dropped": max(len(pool) - len(ranked), 0),
                }
            )
        return ranked
