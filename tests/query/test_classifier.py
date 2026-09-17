from app.query.classifier import classify_query
from tests.query.fakes import DEFAULT_SCRIPTED


def test_pricing_question() -> None:
    assert classify_query("How much does RingCentral cost?", extractor=DEFAULT_SCRIPTED) == "pricing"


def test_comparison_question() -> None:
    assert (
        classify_query("Who is Five9 better for compared with Dialpad?", extractor=DEFAULT_SCRIPTED)
        == "comparison"
    )


def test_review_question() -> None:
    assert (
        classify_query("What do customers say about Nextiva support?", extractor=DEFAULT_SCRIPTED)
        == "review"
    )


def test_multi_hop_two_providers_and_topics() -> None:
    assert (
        classify_query(
            "Compare RingCentral and Nextiva pricing and Salesforce integrations.",
            extractor=DEFAULT_SCRIPTED,
        )
        == "multi-hop"
    )


