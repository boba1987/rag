from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_ENABLED,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)


class NullObservation:
    def update(self, **kwargs: Any) -> None:
        return None


@dataclass
class RecordedSpan:
    name: str
    as_type: str
    input: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    model: str | None = None

    def update(self, **kwargs: Any) -> None:
        if "input" in kwargs:
            self.input = kwargs["input"]
        if "output" in kwargs:
            self.output = kwargs["output"]
        if "metadata" in kwargs:
            self.metadata.update(kwargs["metadata"] or {})
        if "model" in kwargs:
            self.model = kwargs["model"]


class RecordingTracer:
    """In-memory tracer for tests. Does not call Langfuse."""

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self.flushed = False

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[RecordedSpan]:
        recorded = RecordedSpan(
            name=name,
            as_type=as_type,
            input=input,
            metadata=dict(metadata or {}),
            model=model,
        )
        self.spans.append(recorded)
        yield recorded

    def flush(self) -> None:
        self.flushed = True


class LangfuseTracer:
    def __init__(self, client) -> None:
        self._client = client

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[Any]:
        kwargs: dict[str, Any] = {"name": name}
        if input is not None:
            kwargs["input"] = input
        if metadata:
            kwargs["metadata"] = metadata
        if model:
            kwargs["model"] = model
        manager = _start_observation(self._client, as_type=as_type, **kwargs)
        if manager is None:
            yield NullObservation()
            return
        with manager as observation:
            yield observation or NullObservation()

    def flush(self) -> None:
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            flush()


class NullTracer:
    @contextmanager
    def observation(self, name: str, **kwargs: Any) -> Iterator[NullObservation]:
        yield NullObservation()

    def flush(self) -> None:
        return None


_TRACER: RecordingTracer | LangfuseTracer | NullTracer | None = None


def langfuse_enabled() -> bool:
    if LANGFUSE_ENABLED in {"0", "false", "no", "off"}:
        return False
    if LANGFUSE_ENABLED in {"1", "true", "yes", "on"}:
        return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def get_tracer() -> RecordingTracer | LangfuseTracer | NullTracer:
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    if not langfuse_enabled():
        _TRACER = NullTracer()
        return _TRACER
    client = _build_langfuse_client()
    _TRACER = LangfuseTracer(client) if client is not None else NullTracer()
    return _TRACER


def reset_tracer(tracer: RecordingTracer | LangfuseTracer | NullTracer | None = None) -> None:
    global _TRACER
    _TRACER = tracer


def span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
):
    return get_tracer().observation(
        name,
        as_type=as_type,
        input=input,
        metadata=metadata,
        model=model,
    )


def _build_langfuse_client():
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        base_url=LANGFUSE_BASE_URL,
        host=LANGFUSE_BASE_URL,
    )


def _start_observation(client, *, as_type: str, **kwargs: Any):
    if hasattr(client, "start_as_current_observation"):
        return client.start_as_current_observation(as_type=as_type, **kwargs)
    if as_type == "generation" and hasattr(client, "start_as_current_generation"):
        return client.start_as_current_generation(**kwargs)
    if hasattr(client, "start_as_current_span"):
        return client.start_as_current_span(**kwargs)
    return None
