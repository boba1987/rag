import pytest

from app.models.schemas import RetrievedChunk
from types import SimpleNamespace

from app.config import RERANK_PASSAGE_CHARS
from app.observability.tracing import RecordingTracer, reset_tracer
from app.retrieval.reranker import (
    BGEReranker,
    CrossEncoderReranker,
    LexicalReranker,
    OpenAIReranker,
    get_reranker,
)


def _chunk(chunk_id: str, text: str, score: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title="RingCentral",
        section="Integrations",
        heading_path=["RingCentral", "Integrations"],
        text=text,
        score=score,
    )


def test_cross_encoder_orders_by_pair_score() -> None:
    chunks = [
        _chunk("pricing", "RingEX starts at $20.", score=0.9),
        _chunk("salesforce", "RingCentral supports Salesforce.", score=0.2),
    ]

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [10.0 if "Salesforce" in text else 1.0 for _query, text in pairs]

    reranker = CrossEncoderReranker(predict=predict, top_k=5)
    ranked = reranker.rerank("Does RingCentral integrate with Salesforce?", chunks)
    assert [chunk.id for chunk in ranked] == ["salesforce", "pricing"]
    assert ranked[0].score == 10.0
    assert ranked[0].text == "RingCentral supports Salesforce."


def test_cross_encoder_respects_top_k() -> None:
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta"), _chunk("c", "gamma")]

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(pairs) - index) for index, _pair in enumerate(pairs)]

    reranker = CrossEncoderReranker(predict=predict, top_k=5)
    ranked = reranker.rerank("query", chunks, top_k=1)
    assert [chunk.id for chunk in ranked] == ["a"]


def test_empty_candidates_return_empty() -> None:
    reranker = CrossEncoderReranker(predict=lambda pairs: [])
    assert reranker.rerank("query", []) == []


def test_bge_orders_by_pair_score() -> None:
    chunks = [
        _chunk("pricing", "RingEX starts at $20.", score=0.9),
        _chunk("salesforce", "RingCentral supports Salesforce.", score=0.2),
    ]

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [8.0 if "Salesforce" in text else 0.5 for _query, text in pairs]

    reranker = BGEReranker(predict=predict, top_k=5)
    ranked = reranker.rerank("Does RingCentral integrate with Salesforce?", chunks)
    assert [chunk.id for chunk in ranked] == ["salesforce", "pricing"]
    assert ranked[0].score == 8.0


def test_get_reranker_uses_openai_even_for_minilm_aliases(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.reranker.OPENAI_API_KEY", "test-key")
    assert isinstance(get_reranker("openai"), OpenAIReranker)
    assert isinstance(get_reranker("cross-encoder"), OpenAIReranker)
    assert isinstance(get_reranker("bge"), OpenAIReranker)


def test_get_reranker_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("colbert")


def test_lexical_reranker_prefers_overlapping_passage() -> None:
    chunks = [
        _chunk("pricing", "RingEX starts at $20.", score=0.9),
        _chunk("salesforce", "RingCentral supports Salesforce.", score=0.2),
    ]
    ranked = LexicalReranker(top_k=5).rerank(
        "Does RingCentral integrate with Salesforce?",
        chunks,
    )
    assert [chunk.id for chunk in ranked] == ["salesforce", "pricing"]


def test_openai_reranker_orders_by_model_indices() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"order": [1, 0]}')
                    )
                ]
            )

    chunks = [
        _chunk("pricing", "RingEX starts at $20.", score=0.9),
        _chunk("salesforce", "RingCentral supports Salesforce.", score=0.2),
    ]
    ranked = OpenAIReranker(client=_Client(), top_k=5).rerank(
        "Does RingCentral integrate with Salesforce?",
        chunks,
    )
    assert [chunk.id for chunk in ranked] == ["salesforce", "pricing"]
    assert ranked[0].score > ranked[1].score


class _PromptCapturingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.prompts: list[str] = []

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"order": [0]}'))]
        )


def test_passage_chars_truncates_the_prompt() -> None:
    client = _PromptCapturingClient()
    OpenAIReranker(client=client, top_k=1, passage_chars=10).rerank(
        "query",
        [_chunk("long", "x" * 500)],
    )
    assert "x" * 10 in client.prompts[0]
    assert "x" * 11 not in client.prompts[0]


def test_passage_chars_defaults_to_the_configured_value() -> None:
    client = _PromptCapturingClient()
    text = "x" * (RERANK_PASSAGE_CHARS + 100)
    OpenAIReranker(client=client, top_k=1).rerank("query", [_chunk("long", text)])
    assert "x" * RERANK_PASSAGE_CHARS in client.prompts[0]
    assert "x" * (RERANK_PASSAGE_CHARS + 1) not in client.prompts[0]


def test_rerank_llm_span_records_model_and_usage() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"order": [1, 0]}'))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=8, total_tokens=128),
            )

    recorder = RecordingTracer()
    reset_tracer(recorder)
    try:
        OpenAIReranker(client=_Client(), model="gpt-test", top_k=2).rerank(
            "query",
            [_chunk("a", "alpha"), _chunk("b", "beta")],
        )
    finally:
        reset_tracer(None)
    llm = next(item for item in recorder.spans if item.name == "rerank.llm")
    assert llm.as_type == "generation"
    assert llm.model == "gpt-test"
    assert llm.metadata["usage"] == {"input": 120, "output": 8, "total": 128}
    assert llm.output["order"] == [1, 0]


def test_rerank_llm_span_marks_lexical_fallback() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            raise RuntimeError("boom")

    recorder = RecordingTracer()
    reset_tracer(recorder)
    try:
        ranked = OpenAIReranker(client=_Client(), top_k=2).rerank(
            "query",
            [_chunk("a", "alpha")],
        )
    finally:
        reset_tracer(None)
    llm = next(item for item in recorder.spans if item.name == "rerank.llm")
    assert llm.metadata["fallback"] is True
    assert llm.output["fallback"] == "lexical"
    assert [chunk.id for chunk in ranked] == ["a"]


def test_openai_reranker_falls_back_to_lexical_on_bad_json() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
            )

    chunks = [
        _chunk("pricing", "RingEX starts at $20.", score=0.9),
        _chunk("salesforce", "RingCentral supports Salesforce.", score=0.2),
    ]
    ranked = OpenAIReranker(client=_Client(), top_k=5).rerank(
        "Does RingCentral integrate with Salesforce?",
        chunks,
    )
    assert [chunk.id for chunk in ranked] == ["salesforce", "pricing"]
