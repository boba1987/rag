from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import RetrievedChunk


class _FakeRetriever:
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
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


def test_post_query_returns_answer_sources_and_dense_strategy() -> None:
    client = TestClient(create_app(retriever=_FakeRetriever(), generator=_FakeGenerator()))
    response = client.post("/query", json={"query": "Does RingCentral integrate with Salesforce?"})
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "answer": "Yes. RingCentral supports Salesforce.",
        "sources": [
            {
                "title": "RingCentral Review",
                "section": "Integrations",
                "url": "https://example.test/ringcentral",
            }
        ],
        "retrieval": {"strategy": "dense"},
    }


def test_post_query_rejects_empty_query() -> None:
    client = TestClient(create_app(retriever=_FakeRetriever(), generator=_FakeGenerator()))
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
