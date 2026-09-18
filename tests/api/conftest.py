import pytest
from fastapi.testclient import TestClient

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture(autouse=True)
def configure_api_key(monkeypatch) -> None:
    monkeypatch.setattr("app.config.API_KEY", TEST_API_KEY)


class AuthedClient(TestClient):
    def request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("X-API-Key", TEST_API_KEY)
        return super().request(method, url, headers=headers, **kwargs)
