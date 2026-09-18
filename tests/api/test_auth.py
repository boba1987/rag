from fastapi.testclient import TestClient

from app.main import create_app
from tests.api.conftest import AUTH_HEADERS, AuthedClient, TEST_API_KEY
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


def test_docs_and_openapi_require_api_key() -> None:
    raw = TestClient(_app())
    assert raw.get("/docs").status_code == 401
    assert raw.get("/openapi.json").status_code == 401
    authed = AuthedClient(_app())
    assert authed.get("/docs").status_code == 200
    assert authed.get("/openapi.json").status_code == 200


def test_missing_configured_key_returns_service_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.config.API_KEY", "")
    response = TestClient(_app()).post(
        "/query",
        json={"query": "Does RingCentral integrate with Salesforce?"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "API key is not configured"
