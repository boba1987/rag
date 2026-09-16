from math import log2

from app.evaluation.retrieval import (
    mean_retrieval_scores,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
    unique_document_ids,
)
from app.models.schemas import RetrievedChunk


def _chunk(document_id: str, chunk_id: str = "c") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=document_id,
        content_type="provider",
        title="Doc",
        section="Section",
        heading_path=["Section"],
        text="text",
        score=0.9,
    )


def test_recall_precision_mrr_and_ndcg_on_known_ranking() -> None:
    retrieved = ["8019", "8015", "106805"]
    relevant = ["8015"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3
    assert reciprocal_rank(retrieved, relevant) == 0.5
    expected_ndcg = (1 / log2(3)) / (1 / log2(2))
    assert ndcg_at_k(retrieved, relevant, k=3) == expected_ndcg


def test_perfect_ranking_has_ndcg_one() -> None:
    retrieved = ["8019", "8015", "106805"]
    relevant = ["8019", "8015"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3
    assert reciprocal_rank(retrieved, relevant) == 1.0
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0


def test_misses_score_zero() -> None:
    retrieved = ["106805"]
    relevant = ["8019"]
    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert precision_at_k(retrieved, relevant, k=1) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert ndcg_at_k(retrieved, relevant, k=1) == 0.0


def test_empty_relevant_skips_recall_and_ndcg() -> None:
    scores = score_retrieval(["8019"], [], k=5)
    assert scores.recall_at_k is None
    assert scores.ndcg_at_k is None
    assert scores.precision_at_k == 0.0
    assert scores.reciprocal_rank == 0.0


def test_mean_skips_unanswerable_for_recall_and_ndcg() -> None:
    scored = [
        score_retrieval(["8015"], ["8015"], k=5),
        score_retrieval(["8019"], [], k=5),
    ]
    means = mean_retrieval_scores(scored)
    assert means["recall_at_k"] == 1.0
    assert means["ndcg_at_k"] == 1.0
    assert means["mrr"] == 0.5
    assert means["precision_at_k"] == 0.1


def test_unique_document_ids_keep_first_rank() -> None:
    chunks = [
        _chunk("8019", "a"),
        _chunk("8019", "b"),
        _chunk("8015", "c"),
    ]
    assert unique_document_ids(chunks) == ["8019", "8015"]
