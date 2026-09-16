from fastapi.testclient import TestClient

from app.api.query import answer_query
from app.main import create_app
from app.models.schemas import RetrievedChunk


class _FakeRetriever:
    def __init__(self) -> None:
        self.last_filters = None

    def search(self, query: str, top_k: int | None = None, filters=None, infer: bool = False):
        self.last_filters = filters
        return [
            RetrievedChunk(
                id="provider_8015_integrations_01",
                document_id="8015",
                content_type="provider",
                provider="RingCentral",
                title="RingCentral Review",
                section="Integrations",
                heading_path=["RingCentral Review", "Integrations"],
                text="RingCentral supports Salesforce.",
                source_url="https://example.test/ringcentral",
                score=0.91,
            )
        ]


class _FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return "Yes. RingCentral supports Salesforce."


def test_post_query_returns_answer_sources_and_inferred_filters() -> None:
    retriever = _FakeRetriever()
    client = TestClient(create_app(retriever=retriever, generator=_FakeGenerator()))
    response = client.post("/query", json={"query": "Does RingCentral integrate with Salesforce?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Yes. RingCentral supports Salesforce."
    assert body["sources"] == [
        {
            "title": "RingCentral Review",
            "section": "Integrations",
            "url": "https://example.test/ringcentral",
        }
    ]
    assert body["retrieval"]["strategy"] == "dense"
    assert body["retrieval"]["inferred"] is True
    assert body["retrieval"]["filters"]["provider"] == "RingCentral"
    assert body["retrieval"]["filters"]["section"] == "Integration"
    assert retriever.last_filters.provider == "RingCentral"


def test_post_query_explicit_filters_override_inference() -> None:
    retriever = _FakeRetriever()
    client = TestClient(create_app(retriever=retriever, generator=_FakeGenerator()))
    response = client.post(
        "/query",
        json={
            "query": "Does RingCentral integrate with Salesforce?",
            "filters": {"provider": "Nextiva"},
        },
    )
    assert response.status_code == 200
    assert response.json()["retrieval"]["filters"]["provider"] == "Nextiva"
    assert retriever.last_filters.provider == "Nextiva"


def test_post_query_can_disable_inference() -> None:
    retriever = _FakeRetriever()
    client = TestClient(create_app(retriever=retriever, generator=_FakeGenerator()))
    response = client.post(
        "/query",
        json={"query": "Does RingCentral integrate with Salesforce?", "infer": False},
    )
    assert response.status_code == 200
    assert response.json()["retrieval"]["filters"] is None
    assert response.json()["retrieval"]["inferred"] is False
    assert retriever.last_filters is None


def test_post_query_rejects_empty_query() -> None:
    client = TestClient(create_app(retriever=_FakeRetriever(), generator=_FakeGenerator()))
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_post_query_echoes_sparse_and_hybrid_strategy() -> None:
    client = TestClient(create_app(retriever=_FakeRetriever(), generator=_FakeGenerator()))
    for strategy in ("sparse", "hybrid"):
        response = client.post(
            "/query",
            json={
                "query": "Does RingCentral integrate with Salesforce?",
                "strategy": strategy,
            },
        )
        assert response.status_code == 200
        assert response.json()["retrieval"]["strategy"] == strategy


def test_post_query_rejects_unknown_strategy() -> None:
    client = TestClient(create_app(retriever=_FakeRetriever(), generator=_FakeGenerator()))
    response = client.post(
        "/query",
        json={"query": "Does RingCentral integrate with Salesforce?", "strategy": "rerank"},
    )
    assert response.status_code == 422


def test_answer_query_with_no_hits_stays_dense() -> None:
    class _EmptyRetriever:
        def search(self, query: str, top_k: int | None = None, filters=None, infer: bool = False):
            return []

    class _RefuseGenerator:
        def generate(self, question: str, context: str) -> str:
            assert context == ""
            return "I do not know."

    result = answer_query(
        "What is Zoom Phone's 2020 revenue?",
        retriever=_EmptyRetriever(),
        generator=_RefuseGenerator(),
        infer=False,
    )
    assert result.answer == "I do not know."
    assert result.sources == []
    assert result.retrieval.strategy == "dense"
    assert result.retrieval.filters is None
