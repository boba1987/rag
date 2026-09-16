from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import DENSE_TOP_K
from app.evaluation.retrieval import precision_at_k, recall_at_k
from app.models.schemas import EvalCategory

_TOKEN_RE = re.compile(r"[a-z0-9$]+")
_REFUSAL_MARKERS = (
    "do not know",
    "don't know",
    "does not contain",
    "does not have",
    "not in the",
    "no retrieved context",
    "ambiguous",
    "which provider",
    "which plan",
)


@dataclass(frozen=True)
class GenerationScores:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float | None
    groundedness: float


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def token_overlap(numerator: str, denominator: str) -> float:
    """Fraction of numerator tokens that also appear in denominator."""
    left = tokenize(numerator)
    if not left:
        return 0.0
    return len(left & tokenize(denominator)) / len(left)


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def score_generation(
    answer: str,
    context: str,
    retrieved_ids: list[str],
    expected_ids: list[str],
    expected_answer: str | None,
    category: EvalCategory,
    k: int = DENSE_TOP_K,
) -> GenerationScores:
    groundedness = token_overlap(answer, context) if context.strip() else 0.0
    if category in {"unanswerable", "ambiguous"} and looks_like_refusal(answer):
        faithfulness = 1.0
        groundedness = max(groundedness, 1.0)
    else:
        faithfulness = groundedness

    reference = expected_answer or ""
    answer_relevance = token_overlap(answer, reference) if reference else token_overlap(answer, context)
    return GenerationScores(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=precision_at_k(retrieved_ids, expected_ids, k),
        context_recall=recall_at_k(retrieved_ids, expected_ids, k),
        groundedness=groundedness,
    )


def mean_generation_scores(scores: list[GenerationScores]) -> dict[str, float]:
    if not scores:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "groundedness": 0.0,
        }
    recalls = [row.context_recall for row in scores if row.context_recall is not None]
    return {
        "faithfulness": _mean([row.faithfulness for row in scores]),
        "answer_relevance": _mean([row.answer_relevance for row in scores]),
        "context_precision": _mean([row.context_precision for row in scores]),
        "context_recall": _mean(recalls),
        "groundedness": _mean([row.groundedness for row in scores]),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
