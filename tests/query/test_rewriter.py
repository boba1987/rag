from app.query.rewriter import rewrite_query


def test_strips_filler_and_canonicalizes_provider() -> None:
    rewritten = rewrite_query("Can you tell me how much Ring Central costs?")
    assert rewritten.lower().startswith("how much")
    assert "RingCentral" in rewritten
    assert "pricing" in rewritten.lower()
    assert "can you" not in rewritten.lower()


def test_review_query_adds_reviews_term() -> None:
    rewritten = rewrite_query("What do customers say about Nextiva support?")
    assert "Nextiva" in rewritten
    assert "reviews" in rewritten.lower()


def test_comparison_keeps_both_providers() -> None:
    rewritten = rewrite_query("Who is Five9 better for compared with Dialpad?")
    assert "Five9" in rewritten
    assert "Dialpad" in rewritten
    assert "compare" in rewritten.lower()


def test_empty_after_strip_returns_original() -> None:
    assert rewrite_query("   ") == ""
