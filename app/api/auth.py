"""Require API_KEY on every HTTP request."""

from __future__ import annotations

from hmac import compare_digest

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from app import config

_HEADER = "x-api-key"
API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Value of API_KEY from the server environment.",
)


def extract_api_key(request: Request) -> str:
    header = request.headers.get(_HEADER) or ""
    if header:
        return header
    authorization = request.headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def api_keys_match(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    left = provided.encode("utf-8")
    right = expected.encode("utf-8")
    if len(left) != len(right):
        return False
    return compare_digest(left, right)


def is_public_path(path: str) -> bool:
    return path.rstrip("/") in {"/docs", "/redoc", "/openapi.json"} or path == "/docs/oauth2-redirect"


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if is_public_path(request.url.path):
            return await call_next(request)
        expected = config.API_KEY
        if not expected:
            return JSONResponse({"detail": "API key is not configured"}, status_code=503)
        if not api_keys_match(extract_api_key(request), expected):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)
