import os

import pytest


@pytest.fixture(autouse=True)
def heuristic_query_rewriter(monkeypatch) -> None:
    """Keep unit tests offline. Live RAG tests opt in with RUN_LIVE_RAG=1."""
    if os.getenv("RUN_LIVE_RAG") == "1":
        return
    monkeypatch.setattr("app.query.rewriter.QUERY_REWRITER", "heuristic")
