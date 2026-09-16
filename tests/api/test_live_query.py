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
