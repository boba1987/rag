import os

import pytest

from app.api.query import answer_query

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_RAG") != "1",
    reason="Set RUN_LIVE_RAG=1 to hit live Qdrant and OpenAI",
)


def test_live_ringcentral_salesforce_has_sources() -> None:
    result = answer_query("Does RingCentral integrate with Salesforce?")
    assert result.retrieval.strategy == "dense"
    assert result.answer
    assert result.sources
    assert any("RingCentral" in source.title or "Salesforce" in source.section for source in result.sources)


def test_live_ringcentral_cost_filter_keeps_ringcentral_sources() -> None:
    filtered = answer_query("How much does RingCentral cost?", infer=True)
    unfiltered = answer_query("How much does RingCentral cost?", infer=False)
    assert filtered.retrieval.filters is not None
    assert filtered.retrieval.filters.provider == "RingCentral"
    assert filtered.retrieval.filters.section
    assert filtered.sources
    assert all("RingCentral" in source.title for source in filtered.sources)
    assert unfiltered.retrieval.filters is None
    assert unfiltered.sources
