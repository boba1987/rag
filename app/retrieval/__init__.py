from app.config import ACTIVE_CHUNKER
from app.models.schemas import RetrievalStrategy
from app.retrieval.dense import DenseRetriever
from app.retrieval.filters import build_qdrant_filter, infer_filters, merge_filters
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.parent_child import (
    FileParentStore,
    ParentExpandingRetriever,
    expand_to_parents,
)
from app.retrieval.reranker import (
    BGEReranker,
    CrossEncoderReranker,
    OpenAIReranker,
    RerankRetriever,
    Reranker,
    get_reranker,
)
from app.retrieval.sparse import SparseRetriever, load_chunk_corpus


def retriever_for(strategy: RetrievalStrategy = "dense"):
    if strategy == "dense":
        retriever = DenseRetriever()
    elif strategy == "sparse":
        retriever = SparseRetriever()
    elif strategy == "hybrid":
        retriever = HybridRetriever()
    elif strategy == "rerank":
        retriever = RerankRetriever()
    elif strategy == "raptor":
        from app.raptor.retrieval import RaptorRetriever

        return RaptorRetriever()
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    if ACTIVE_CHUNKER == "parent_child":
        return ParentExpandingRetriever(retriever)
    return retriever


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
    "OpenAIReranker",
    "BGEReranker",
    "RerankRetriever",
    "get_reranker",
    "FileParentStore",
    "ParentExpandingRetriever",
    "expand_to_parents",
]
