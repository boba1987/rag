from app.query.classifier import classify_query


def test_pricing_question() -> None:
    assert classify_query("How much does RingCentral cost?") == "pricing"


def test_comparison_question() -> None:
    assert classify_query("Who is Five9 better for compared with Dialpad?") == "comparison"


def test_recommendation_question() -> None:
    assert classify_query("Is RingCentral a good fit for a 20-person sales team?") == "recommendation"


def test_review_question() -> None:
    assert classify_query("What do customers say about Nextiva support?") == "review"


def test_multi_hop_two_providers_and_topics() -> None:
    assert (
        classify_query("Compare RingCentral and Nextiva pricing and Salesforce integrations.")
        == "multi-hop"
    )


def test_factual_integration_question() -> None:
    assert classify_query("Does RingCentral integrate with Salesforce?") == "factual"
