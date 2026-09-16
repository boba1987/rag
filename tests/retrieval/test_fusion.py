from app.models.schemas import RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion


def _chunk(chunk_id: str, document_id: str = "8015", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=document_id,
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Integrations",
        heading_path=["RingCentral", "Integrations"],
        text="RingCentral supports Salesforce.",
        score=score,
    )


def test_chunk_in_both_lists_outranks_single_list_hit() -> None:
    dense = [_chunk("a", score=0.9), _chunk("b", score=0.8)]
    sparse = [_chunk("a", score=4.0), _chunk("c", score=3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert [chunk.id for chunk in fused] == ["a", "b", "c"]
    assert fused[0].score == 2 / 61
    assert fused[1].score == 1 / 62
    assert fused[2].score == 1 / 62


def test_empty_rankings_return_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order() -> None:
    ranking = [_chunk("b"), _chunk("a")]
    fused = reciprocal_rank_fusion([ranking], k=60)
    assert [chunk.id for chunk in fused] == ["b", "a"]
