from __future__ import annotations

from dataclasses import dataclass
from math import log2

from app.config import DENSE_TOP_K
from app.models.schemas import RetrievedChunk


@dataclass(frozen=True)
class RetrievalScores:
    k: int
    recall_at_k: float | None
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float | None


def unique_document_ids(chunks: list[RetrievedChunk]) -> list[str]:
    """Preserve first-seen document ids from a ranked chunk list."""
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.document_id in seen:
            continue
        seen.add(chunk.document_id)
        ids.append(chunk.document_id)
    return ids


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant:
        return None
    hits = set(retrieved[:k]) & set(relevant)
    return len(hits) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = set(retrieved[:k]) & set(relevant)
    return len(hits) / k


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for index, document_id in enumerate(retrieved, start=1):
        if document_id in relevant_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant:
        return None
    relevant_set = set(relevant)
    dcg = sum(_gain(index) for index, document_id in enumerate(retrieved[:k], start=1) if document_id in relevant_set)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(_gain(index) for index in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def score_retrieval(
    retrieved: list[str],
    relevant: list[str],
    k: int = DENSE_TOP_K,
) -> RetrievalScores:
    return RetrievalScores(
        k=k,
        recall_at_k=recall_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
    )


def mean_retrieval_scores(scores: list[RetrievalScores]) -> dict[str, float]:
    """Average scored queries. Recall and nDCG skip cases with no expected documents."""
    if not scores:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }
    recalls = [row.recall_at_k for row in scores if row.recall_at_k is not None]
    ndcgs = [row.ndcg_at_k for row in scores if row.ndcg_at_k is not None]
    return {
        "recall_at_k": _mean(recalls),
        "precision_at_k": _mean([row.precision_at_k for row in scores]),
        "mrr": _mean([row.reciprocal_rank for row in scores]),
        "ndcg_at_k": _mean(ndcgs),
    }


def _gain(rank: int) -> float:
    return 1.0 / log2(rank + 1)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
