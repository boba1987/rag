import pytest

from app.models.schemas import RetrievedChunk
from app.retrieval.reranker import BGEReranker, CrossEncoderReranker, get_reranker


def _chunk(chunk_id: str, text: str, score: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Integrations",
        heading_path=["RingCentral", "Integrations"],
        text=text,
        score=score,
    )


def test_cross_encoder_orders_by_pair_score() -> None:
    chunks = [
        _chunk("pricing", "RingEX starts at $20.", score=0.9),
        _chunk("salesforce", "RingCentral supports Salesforce.", score=0.2),
    ]

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [10.0 if "Salesforce" in text else 1.0 for _query, text in pairs]

    reranker = CrossEncoderReranker(predict=predict, top_k=5)
    ranked = reranker.rerank("Does RingCentral integrate with Salesforce?", chunks)
    assert [chunk.id for chunk in ranked] == ["salesforce", "pricing"]
    assert ranked[0].score == 10.0
    assert ranked[0].text == "RingCentral supports Salesforce."


def test_cross_encoder_respects_top_k() -> None:
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta"), _chunk("c", "gamma")]

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(pairs) - index) for index, _pair in enumerate(pairs)]

    reranker = CrossEncoderReranker(predict=predict, top_k=5)
    ranked = reranker.rerank("query", chunks, top_k=1)
    assert [chunk.id for chunk in ranked] == ["a"]


def test_empty_candidates_return_empty() -> None:
    reranker = CrossEncoderReranker(predict=lambda pairs: [])
    assert reranker.rerank("query", []) == []


def test_bge_placeholder_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="BGE reranker"):
        BGEReranker().rerank("query", [_chunk("a", "alpha")])


def test_get_reranker_selects_cross_encoder() -> None:
    assert isinstance(get_reranker("cross-encoder"), CrossEncoderReranker)


def test_get_reranker_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("colbert")
