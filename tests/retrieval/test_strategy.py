import pytest

from app.retrieval import SparseRetriever, retriever_for


def test_retriever_for_sparse_is_bm25() -> None:
    assert isinstance(retriever_for("sparse"), SparseRetriever)


def test_retriever_for_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        retriever_for("rerank")  # type: ignore[arg-type]
