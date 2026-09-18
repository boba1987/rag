from fastapi.testclient import TestClient

from app.main import create_app
from tests.api.conftest import AUTH_HEADERS, TEST_API_KEY
from tests.query.fakes import DEFAULT_SCRIPTED


class _FakeRetriever:
    def search(self, query: str, top_k: int | None = None, filters=None, infer: bool = False):
        return []


class _FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return "ok"


def _app():
    return create_app(retriever=_FakeRetriever(), generator=_FakeGenerator(), extractor=DEFAULT_SCRIPTED)


def test_query_without_api_key_is_unauthorized() -> None:
    response = TestClient(_app()).post("/query", json={"query": "Does RingCentral integrate with Salesforce?"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_query_with_wrong_api_key_is_unauthorized() -> None:
    response = TestClient(_app()).post(
        "/query",
        json={"query": "Does RingCentral integrate with Salesforce?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_query_accepts_bearer_api_key() -> None:
    client = TestClient(_app())
    response = client.post(
        "/query",
        json={"query": "Does RingCentral integrate with Salesforce?"},
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert response.status_code == 200


def test_docs_and_openapi_are_public() -> None:
    raw = TestClient(_app())
    assert raw.get("/docs").status_code == 200
    spec = raw.get("/openapi.json")
    assert spec.status_code == 200
    payload = spec.json()
    example = payload["components"]["schemas"]["QueryRequest"]["example"]
    assert example == {
        "query": "who is better for startups nextiva or dialpad?",
        "strategy": "rerank",
    }
    scheme = payload["components"]["securitySchemes"]["APIKeyHeader"]
    assert scheme == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Value of API_KEY from the server environment.",
    }
    assert {"APIKeyHeader": []} in payload["paths"]["/query"]["post"].get("security", [])
    assert raw.get("/redoc").status_code == 200
    assert raw.post("/query", json={"query": "Does RingCentral integrate with Salesforce?"}).status_code == 401


def test_missing_configured_key_returns_service_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.config.API_KEY", "")
    response = TestClient(_app()).post(
        "/query",
        json={"query": "Does RingCentral integrate with Salesforce?"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "API key is not configured"
