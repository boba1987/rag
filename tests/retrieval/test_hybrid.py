from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.retrieval.hybrid import HybridRetriever


def _chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Integrations",
        heading_path=["RingCentral", "Integrations"],
        text="RingCentral supports Salesforce.",
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


def test_overlap_outranks_single_list_hits() -> None:
    dense = _FixedRetriever([_chunk("a", 0.9), _chunk("b", 0.8)])
    sparse = _FixedRetriever([_chunk("a", 4.0), _chunk("c", 3.0)])
    retriever = HybridRetriever(dense=dense, sparse=sparse, top_k=5, rrf_k=60)
    results = retriever.search("Salesforce")
    assert [chunk.id for chunk in results] == ["a", "b", "c"]
    assert results[0].score == 2 / 61


def test_forwards_filters_and_infer_to_both() -> None:
    dense = _FixedRetriever([_chunk("a")])
    sparse = _FixedRetriever([_chunk("b")])
    retriever = HybridRetriever(dense=dense, sparse=sparse, top_k=3)
    filters = RetrievalFilters(provider="RingCentral")
    retriever.search("Salesforce", top_k=2, filters=filters, infer=True)
    for calls in (dense.calls, sparse.calls):
        assert calls == [
            {
                "query": "Salesforce",
                "top_k": 2,
                "filters": filters,
                "infer": True,
            }
        ]


def test_respects_top_k_after_fusion() -> None:
    dense = _FixedRetriever([_chunk("a"), _chunk("b")])
    sparse = _FixedRetriever([_chunk("c"), _chunk("d")])
    retriever = HybridRetriever(dense=dense, sparse=sparse, top_k=5, rrf_k=60)
    results = retriever.search("Salesforce", top_k=1)
    assert [chunk.id for chunk in results] == ["a"]


def test_empty_sides_return_empty() -> None:
    retriever = HybridRetriever(
        dense=_FixedRetriever([]),
        sparse=_FixedRetriever([]),
    )
    assert retriever.search("Salesforce") == []
