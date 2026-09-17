from qdrant_client.http.models import Filter, MatchValue

from app.models.schemas import RetrievalFilters
from app.query.extractor import QueryExtraction
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


def test_infers_from_extraction() -> None:
    filters = infer_filters(
        "How much does RingCentral cost?",
        QueryExtraction(kind="pricing", providers=["RingCentral"], topics=["pricing"], source="openai"),
    )
    assert filters.provider == "RingCentral"
    assert filters.section == "pricing"
    assert filters.content_type == "provider"


def test_infers_review_content_type() -> None:
    filters = infer_filters(
        "What do customers say about Nextiva support?",
        QueryExtraction(kind="review", providers=["Nextiva"], topics=["support"], source="openai"),
    )
    assert filters.provider == "Nextiva"
    assert filters.section == "support"
    assert filters.content_type == "review"


def test_without_extraction_only_matches_catalog_providers(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.query.catalog import catalog_from_qdrant

    class _Client:
        def collection_exists(self, name: str) -> bool:
            return True

        def scroll(self, **kwargs):
            return [
                SimpleNamespace(
                    payload={
                        "provider": "RingCentral",
                        "title": "RingCentral",
                        "section": "Pricing",
                        "text": "",
                    }
                )
            ], None

    monkeypatch.setattr("app.retrieval.filters.get_catalog", lambda: catalog_from_qdrant(client=_Client()))
    filters = infer_filters("How much does RingCentral cost?")
    assert filters.provider == "RingCentral"
    assert filters.section is None


def test_does_not_invent_unknown_providers() -> None:
    filters = infer_filters("How much does AcmePBX cost per concurrent user?")
    assert filters.provider is None


def test_two_catalog_providers_do_not_force_a_single_filter() -> None:
    filters = infer_filters(
        "Compare RingCentral and Nextiva pricing.",
        QueryExtraction(
            kind="comparison",
            providers=["RingCentral", "Nextiva"],
            topics=["pricing"],
            source="openai",
        ),
    )
    assert filters.provider is None
    assert filters.section == "pricing"


def test_explicit_filters_override_inferred() -> None:
    merged = merge_filters(
        RetrievalFilters(provider="Nextiva"),
        infer_filters(
            "How much does RingCentral cost?",
            QueryExtraction(kind="pricing", providers=["RingCentral"], topics=["pricing"], source="openai"),
        ),
    )
    assert merged is not None
    assert merged.provider == "Nextiva"
    assert merged.section == "pricing"
