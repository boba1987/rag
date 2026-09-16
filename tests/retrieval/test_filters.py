from qdrant_client.http.models import Filter, MatchValue

from app.models.schemas import RetrievalFilters
from app.retrieval.filters import build_qdrant_filter


def test_empty_filters_are_none() -> None:
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter(RetrievalFilters()) is None


def test_builds_must_conditions_for_set_fields() -> None:
    qfilter = build_qdrant_filter(
        RetrievalFilters(provider="RingCentral", section="Pricing", content_type="provider")
    )
    assert isinstance(qfilter, Filter)
    assert qfilter.must is not None
    assert len(qfilter.must) == 3
    keys = {condition.key: condition.match.value for condition in qfilter.must}
    assert keys == {
        "provider": "RingCentral",
        "section": "Pricing",
        "content_type": "provider",
    }
    assert isinstance(qfilter.must[0].match, MatchValue)
