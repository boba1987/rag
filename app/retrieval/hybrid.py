from __future__ import annotations

from app.config import DENSE_TOP_K, RRF_K
from app.models.schemas import RetrievalFilters, RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Run dense and sparse search, then fuse ranks with RRF."""

    def __init__(
        self,
        dense=None,
        sparse=None,
        top_k: int = DENSE_TOP_K,
        rrf_k: int = RRF_K,
    ) -> None:
        self._dense = dense or DenseRetriever()
        self._sparse = sparse or SparseRetriever()
        self._top_k = top_k
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        infer: bool = False,
    ) -> list[RetrievedChunk]:
        limit = top_k or self._top_k
        dense_hits = self._dense.search(
            query, top_k=limit, filters=filters, infer=infer
        )
        sparse_hits = self._sparse.search(
            query, top_k=limit, filters=filters, infer=infer
        )
        return reciprocal_rank_fusion([dense_hits, sparse_hits], k=self._rrf_k)[:limit]
