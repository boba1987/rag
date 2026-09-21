import pytest

from app.observability.tracing import RecordingTracer, langfuse_enabled, reset_tracer
from app.workflows.rag_graph import run_rag_graph
from tests.query.fakes import DEFAULT_SCRIPTED
from tests.workflows.test_rag_graph import _BoomGenerator, _EmptyRetriever, _FakeGenerator, _FakeRetriever


@pytest.fixture
def tracer():
    recorder = RecordingTracer()
    reset_tracer(recorder)
    yield recorder
    reset_tracer(None)


def test_langfuse_stays_off_during_pytest(monkeypatch) -> None:
    monkeypatch.setattr("app.observability.tracing.LANGFUSE_ENABLED", "")
    monkeypatch.setattr("app.observability.tracing.LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr("app.observability.tracing.LANGFUSE_SECRET_KEY", "sk-test")
    assert langfuse_enabled() is False


def test_graph_records_query_spans(tracer: RecordingTracer) -> None:
    result = run_rag_graph(
        "Does RingCentral integrate with Salesforce?",
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        extractor=DEFAULT_SCRIPTED,
        infer=False,
    )
    names = [item.name for item in tracer.spans]
    assert names == [
        "rag.query",
        "query.classify",
        "query.rewrite",
        "retrieval",
        "retrieval.attempt",
        "evidence.check",
        "generation",
    ]
    assert tracer.spans[0].output["answer"] == result.answer
    assert tracer.spans[3].metadata["strategy"] == "dense"
    assert tracer.spans[4].output["chunk_ids"]
    assert tracer.spans[5].metadata["attempt"] == 1
    assert tracer.spans[5].output["sufficient"] is True
    assert tracer.spans[6].as_type == "generation"
    assert tracer.flushed is True


def test_graph_records_abstain_span(tracer: RecordingTracer) -> None:
    run_rag_graph(
        "What is Zoom Phone's 2020 revenue?",
        retriever=_EmptyRetriever(),
        generator=_BoomGenerator(),
        extractor=DEFAULT_SCRIPTED,
        infer=False,
    )
    names = [item.name for item in tracer.spans]
    assert "abstain" in names
    assert "generation" not in names
    assert tracer.spans[-1].name == "abstain" or names[-1] == "rag.query"
    assert any(item.name == "evidence.check" and item.output["sufficient"] is False for item in tracer.spans)


def test_retry_records_both_attempts(tracer: RecordingTracer) -> None:
    run_rag_graph(
        "What is Zoom Phone's 2020 revenue?",
        retriever=_EmptyRetriever(),
        generator=_BoomGenerator(),
        extractor=DEFAULT_SCRIPTED,
        infer=False,
    )
    names = [item.name for item in tracer.spans]
    assert "retrieval.attempt" in names
    assert "retrieval.retry" in names
    attempts = [item.metadata["attempt"] for item in tracer.spans if item.name == "evidence.check"]
    assert attempts == [1, 2]
    retry = next(item for item in tracer.spans if item.name == "retrieval.retry")
    first = next(item for item in tracer.spans if item.name == "retrieval.attempt")
    assert retry.input["queries"] != first.input["queries"]
    retrieval = next(item for item in tracer.spans if item.name == "retrieval")
    assert retrieval.output["retried"] is True
