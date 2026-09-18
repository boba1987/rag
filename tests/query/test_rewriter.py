from types import SimpleNamespace

import pytest

from app.query.catalog import QueryCatalog
from app.query.rewriter import (
    HeuristicRewriter,
    OpenAIRewriter,
    constrain_rewrite,
    get_rewriter,
    rewrite_query,
)
from tests.query.fakes import COMPARISON, PRICING, REVIEW


def _catalog() -> QueryCatalog:
    return QueryCatalog(
        providers=("Nextiva", "RingCentral", "Five9", "Dialpad"),
        aliases=(
            ("ring central", "RingCentral"),
            ("ringcentral", "RingCentral"),
            ("nextiva", "Nextiva"),
            ("five9", "Five9"),
            ("dialpad", "Dialpad"),
        ),
    )


def test_strips_filler_and_canonicalizes_provider() -> None:
    rewritten = rewrite_query(
        "Can you tell me how much Ring Central costs?",
        extraction=PRICING,
        catalog=_catalog(),
        rewriter=HeuristicRewriter(),
    )
    assert rewritten.lower().startswith("how much")
    assert "RingCentral" in rewritten
    assert "pricing" in rewritten.lower()
    assert "can you" not in rewritten.lower()


def test_review_query_adds_reviews_term() -> None:
    rewritten = rewrite_query(
        "What do customers say about Nextiva support?",
        extraction=REVIEW,
        catalog=_catalog(),
        rewriter=HeuristicRewriter(),
    )
    assert "Nextiva" in rewritten
    assert "reviews" in rewritten.lower()


def test_comparison_keeps_both_providers() -> None:
    rewritten = rewrite_query(
        "Who is Five9 better for compared with Dialpad?",
        extraction=COMPARISON,
        catalog=_catalog(),
        rewriter=HeuristicRewriter(),
    )
    assert "Five9" in rewritten
    assert "Dialpad" in rewritten
    assert "compare" in rewritten.lower()


def test_empty_after_strip_returns_original() -> None:
    assert rewrite_query("   ", catalog=_catalog(), rewriter=HeuristicRewriter()) == ""


def test_openai_rewriter_returns_single_query() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"query": "RingCentral pricing plans"}')
                    )
                ]
            )

    rewritten = OpenAIRewriter(client=_Client()).rewrite(
        "Can you tell me how much RingCentral costs?",
        extraction=PRICING,
        catalog=_catalog(),
    )
    assert rewritten == "RingCentral pricing plans"


def test_openai_rewriter_does_not_invent_catalog_provider() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"query": "Nextiva pricing"}')
                    )
                ]
            )

    rewritten = OpenAIRewriter(client=_Client()).rewrite(
        "what is NICE CXone pricing",
        extraction=PRICING,
        catalog=_catalog(),
    )
    assert "Nextiva" not in rewritten
    assert "pricing" in rewritten.lower()


def test_openai_rewriter_keeps_named_providers_if_model_drops_them() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"query": "startup comparison"}'))
                ]
            )

    rewritten = OpenAIRewriter(client=_Client()).rewrite(
        "Who is Five9 better for compared with Dialpad?",
        extraction=COMPARISON,
        catalog=_catalog(),
    )
    assert "Five9" in rewritten
    assert "Dialpad" in rewritten


def test_openai_rewriter_falls_back_when_llm_fails() -> None:
    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("down")

    class _Boom:
        chat = SimpleNamespace(completions=_Completions())

    rewritten = OpenAIRewriter(client=_Boom()).rewrite(
        "Can you tell me how much Ring Central costs?",
        extraction=PRICING,
        catalog=_catalog(),
    )
    assert "RingCentral" in rewritten
    assert "pricing" in rewritten.lower()


def test_constrain_rewrite_rejects_invented_provider() -> None:
    with pytest.raises(ValueError, match="invented providers"):
        constrain_rewrite("what is NICE CXone pricing", "Nextiva pricing", _catalog())


def test_openai_rewriter_uses_nano_model_by_default() -> None:
    class _Completions:
        def __init__(self) -> None:
            self.model = None

        def create(self, **kwargs):
            self.model = kwargs["model"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"query": "hello"}'))]
            )

    completions = _Completions()

    class _Client:
        chat = SimpleNamespace(completions=completions)

    OpenAIRewriter(client=_Client()).rewrite("hello", catalog=_catalog())
    assert completions.model == "gpt-4.1-nano"


def test_get_rewriter_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown query rewriter"):
        get_rewriter("hyde")
