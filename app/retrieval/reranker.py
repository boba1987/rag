from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import (
    BGE_RERANKER_MODEL,
    CROSS_ENCODER_MODEL,
    RERANK_CANDIDATES,
    RERANK_TOP_K,
    RERANKER_PROVIDER,
)
from app.models.schemas import RetrievalFilters, RetrievedChunk


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
    ) -> None:
        self._predict = predict
        self._model_name = model
        self._model = None
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
        scores = self._score_pairs([(query, chunk.text) for chunk in chunks])
        ranked = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return ranked[:limit]

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._predict is not None:
            return [float(score) for score in self._predict(pairs)]
        model = self._load_model()
        return [float(score) for score in model.predict(pairs)]

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install 'rag-service[rerank]'"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model


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
    if name in {"cross-encoder", "cross_encoder"}:
        return CrossEncoderReranker()
    if name == "bge":
        return BGEReranker()
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
        return self._reranker.rerank(query, pool, top_k=limit)
