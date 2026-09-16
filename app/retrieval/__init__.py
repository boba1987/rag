from app.models.schemas import RetrievalStrategy
from app.retrieval.dense import DenseRetriever
from app.retrieval.filters import build_qdrant_filter, infer_filters, merge_filters
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import (
    BGEReranker,
    CrossEncoderReranker,
    RerankRetriever,
    Reranker,
    get_reranker,
)
from app.retrieval.sparse import SparseRetriever, load_chunk_corpus


def retriever_for(strategy: RetrievalStrategy = "dense"):
    if strategy == "dense":
        return DenseRetriever()
    if strategy == "sparse":
        return SparseRetriever()
    if strategy == "hybrid":
        return HybridRetriever()
    raise ValueError(f"Unknown retrieval strategy: {strategy}")


__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "SparseRetriever",
    "build_qdrant_filter",
    "infer_filters",
    "load_chunk_corpus",
    "merge_filters",
    "reciprocal_rank_fusion",
    "retriever_for",
    "Reranker",
    "CrossEncoderReranker",
    "BGEReranker",
    "RerankRetriever",
    "get_reranker",
]
