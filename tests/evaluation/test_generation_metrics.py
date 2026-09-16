from app.evaluation.generation import looks_like_refusal, score_generation, token_overlap


def test_token_overlap_is_fraction_of_answer_in_context() -> None:
    assert token_overlap("Nextiva Core is $15", "Nextiva Core is $15–$23 per user") == 1.0
    assert token_overlap("Zoom revenue 2019", "Nextiva Core pricing") == 0.0


def test_unanswerable_refusal_is_faithful() -> None:
    scores = score_generation(
        answer="The indexed GetVoIP corpus does not contain Zoom Phone's 2020 annual revenue.",
        context="",
        retrieved_ids=["106730"],
        expected_ids=[],
        expected_answer="The indexed GetVoIP corpus does not contain Zoom Phone's 2020 annual revenue.",
        category="unanswerable",
        k=5,
    )
    assert looks_like_refusal(
        "The indexed GetVoIP corpus does not contain Zoom Phone's 2020 annual revenue."
    )
    assert scores.faithfulness == 1.0
    assert scores.groundedness == 1.0
    assert scores.context_recall is None
    assert scores.answer_relevance > 0.5


def test_grounded_answer_scores_overlap_and_context_recall() -> None:
    scores = score_generation(
        answer="Nextiva Core is $15 per user.",
        context="Nextiva Core is $15–$23 per user per month.",
        retrieved_ids=["8019", "8015"],
        expected_ids=["8019"],
        expected_answer="Nextiva Core is $15–$23 per user per month.",
        category="pricing",
        k=5,
    )
    assert scores.groundedness == 1.0
    assert scores.faithfulness == 1.0
    assert scores.context_precision == 0.2
    assert scores.context_recall == 1.0
    assert scores.answer_relevance > 0.4
