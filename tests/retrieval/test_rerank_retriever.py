from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.retrieval.reranker import RerankRetriever


def _chunk(chunk_id: str, score: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Integrations",
        heading_path=["RingCentral", "Integrations"],
        text=chunk_id,
        score=score,
    )


class _FixedRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def search(self, query: str, top_k: int | None = None, filters=None, infer: bool = False):
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters, "infer": infer}
        )
        return self._results[:top_k] if top_k is not None else self._results


class _FixedReranker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None):
        self.calls.append({"query": query, "ids": [chunk.id for chunk in chunks], "top_k": top_k})
        ranked = list(reversed(chunks))
        return ranked[:top_k] if top_k is not None else ranked


def test_hybrid_pool_then_rerank_top_k() -> None:
    retriever = _FixedRetriever([_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")])
    reranker = _FixedReranker()
    pipeline = RerankRetriever(
        retriever=retriever,
        reranker=reranker,
        candidates=3,
        top_k=2,
    )
    results = pipeline.search("Salesforce")
    assert retriever.calls[0]["top_k"] == 3
    assert reranker.calls[0]["ids"] == ["a", "b", "c"]
    assert reranker.calls[0]["top_k"] == 2
    assert [chunk.id for chunk in results] == ["c", "b"]


def test_forwards_filters_and_infer() -> None:
    retriever = _FixedRetriever([_chunk("a")])
    reranker = _FixedReranker()
    filters = RetrievalFilters(provider="RingCentral")
    pipeline = RerankRetriever(retriever=retriever, reranker=reranker, candidates=20, top_k=5)
    pipeline.search("Salesforce", top_k=4, filters=filters, infer=True)
    assert retriever.calls == [
        {"query": "Salesforce", "top_k": 20, "filters": filters, "infer": True}
    ]
    assert reranker.calls[0]["top_k"] == 4


def test_empty_pool_returns_empty() -> None:
    pipeline = RerankRetriever(
        retriever=_FixedRetriever([]),
        reranker=_FixedReranker(),
    )
    assert pipeline.search("Salesforce") == []
