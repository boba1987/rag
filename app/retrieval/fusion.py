from __future__ import annotations

from app.config import RRF_K
from app.models.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Combine ranked lists with RRF: sum 1 / (k + rank). Ties keep first-seen order."""
    fused: dict[str, tuple[float, int, RetrievedChunk]] = {}
    seen = 0
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            score = 1.0 / (k + rank)
            if chunk.id not in fused:
                fused[chunk.id] = (score, seen, chunk)
                seen += 1
            else:
                previous, order, original = fused[chunk.id]
                fused[chunk.id] = (previous + score, order, original)
    ordered = sorted(fused.values(), key=lambda item: (-item[0], item[1]))
    return [chunk.model_copy(update={"score": score}) for score, _order, chunk in ordered]
