from qdrant_client.http.models import Filter, MatchValue

from app.models.schemas import RetrievalFilters
from app.retrieval.filters import build_qdrant_filter, infer_filters, merge_filters


def test_empty_filters_are_none() -> None:
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter(RetrievalFilters()) is None
    assert build_qdrant_filter(RetrievalFilters(section="Pricing")) is None


def test_builds_must_conditions_for_exact_payload_fields() -> None:
    qfilter = build_qdrant_filter(
        RetrievalFilters(provider="RingCentral", section="Pricing", content_type="provider")
    )
    assert isinstance(qfilter, Filter)
    assert qfilter.must is not None
    keys = {condition.key: condition.match.value for condition in qfilter.must}
    assert keys == {"provider": "RingCentral", "content_type": "provider"}
    assert "section" not in keys
    assert isinstance(qfilter.must[0].match, MatchValue)


def test_infers_provider_and_pricing_section() -> None:
    filters = infer_filters("How much does RingCentral cost?")
    assert filters.provider == "RingCentral"
    assert filters.section == "Pricing"
    assert filters.content_type == "provider"


def test_infers_nextiva_integrations() -> None:
    filters = infer_filters("Does Nextiva Core include Google Workspace and Microsoft 365 integrations?")
    assert filters.provider == "Nextiva"
    assert filters.section == "Integration"
    assert filters.content_type == "provider"


def test_infers_review_content_type() -> None:
    filters = infer_filters("What do customers say about Nextiva support?")
    assert filters.provider == "Nextiva"
    assert filters.section == "Support"
    assert filters.content_type == "review"


def test_does_not_invent_unknown_providers() -> None:
    filters = infer_filters("How much does Five9 cost per concurrent user?")
    assert filters.provider is None
    assert filters.section == "Pricing"


def test_ambiguous_cost_query_has_no_provider() -> None:
    filters = infer_filters("How much does it cost?")
    assert filters.provider is None
    assert filters.section == "Pricing"


def test_explicit_filters_override_inferred() -> None:
    merged = merge_filters(
        RetrievalFilters(provider="Nextiva"),
        infer_filters("How much does RingCentral cost?"),
    )
    assert merged is not None
    assert merged.provider == "Nextiva"
    assert merged.section == "Pricing"
