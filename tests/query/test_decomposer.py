from app.query.decomposer import decompose_query, expand_queries
from tests.query.fakes import DEFAULT_SCRIPTED, MULTI_HOP, PRICING


def test_decomposes_pricing_and_salesforce_for_two_providers() -> None:
    parts = decompose_query(
        "Compare RingCentral and Nextiva pricing and Salesforce integrations.",
        extraction=MULTI_HOP,
    )
    assert parts == [
        "RingCentral pricing",
        "Nextiva pricing",
        "RingCentral Salesforce integration",
        "Nextiva Salesforce integration",
    ]


def test_single_provider_does_not_decompose() -> None:
    parts = decompose_query("How much does RingCentral cost?", extraction=PRICING)
    assert len(parts) == 1
    assert "RingCentral" in parts[0]


def test_expand_queries_uses_decomposition_when_split() -> None:
    queries = expand_queries(
        "Compare RingCentral and Nextiva pricing and Salesforce integrations.",
        extraction=MULTI_HOP,
    )
    assert len(queries) == 4
    assert "Nextiva pricing" in queries


def test_expand_queries_adds_pricing_alternate() -> None:
    queries = expand_queries("How much does RingCentral cost?", extraction=PRICING)
    assert len(queries) >= 2
    assert any("per user" in query.lower() for query in queries)


def test_expand_queries_can_use_scripted_extractor() -> None:
    queries = expand_queries(
        "Compare RingCentral and Nextiva pricing and Salesforce integrations.",
        extractor=DEFAULT_SCRIPTED,
    )
    assert len(queries) == 4
