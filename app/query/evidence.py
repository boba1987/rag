from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import BaseModel

from app.config import (
    EVIDENCE_CHECKER,
    EVIDENCE_MIN_OVERLAP,
    OPENAI_API_KEY,
    QUERY_EXTRACTOR_MODEL,
)
from app.models.schemas import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


class EvidenceVerdict(BaseModel):
    sufficient: bool
    reason: str
    overlap: float = 0.0


class EvidenceChecker(ABC):
    name: str

    @abstractmethod
    def check(self, question: str, chunks: list[RetrievedChunk]) -> EvidenceVerdict:
        raise NotImplementedError


class HeuristicEvidenceChecker(EvidenceChecker):
    """Lexical overlap between the question and retrieved passages. No LLM."""

    name = "heuristic"

    def __init__(self, min_overlap: float = EVIDENCE_MIN_OVERLAP) -> None:
        self._min_overlap = min_overlap

    def check(self, question: str, chunks: list[RetrievedChunk]) -> EvidenceVerdict:
        if not chunks:
            return EvidenceVerdict(sufficient=False, reason="no_hits", overlap=0.0)
        overlap = content_overlap(question, chunks)
        if overlap < self._min_overlap:
            return EvidenceVerdict(sufficient=False, reason="low_overlap", overlap=overlap)
        return EvidenceVerdict(sufficient=True, reason="ok", overlap=overlap)


class OpenAIEvidenceChecker(EvidenceChecker):
    name = "openai"

    def __init__(
        self,
        client=None,
        model: str = QUERY_EXTRACTOR_MODEL,
        api_key: str | None = OPENAI_API_KEY,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._fallback = HeuristicEvidenceChecker()

    def check(self, question: str, chunks: list[RetrievedChunk]) -> EvidenceVerdict:
        if not chunks:
            return EvidenceVerdict(sufficient=False, reason="no_hits", overlap=0.0)
        try:
            payload = json.loads(self._complete(question, chunks))
            return EvidenceVerdict(
                sufficient=bool(payload.get("sufficient")),
                reason=str(payload.get("reason") or "ok"),
                overlap=content_overlap(question, chunks),
            )
        except Exception:
            return self._fallback.check(question, chunks)

    def _complete(self, question: str, chunks: list[RetrievedChunk]) -> str:
        passages = "\n\n".join(f"- {chunk.title}: {chunk.text}" for chunk in chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide if the passages contain enough evidence to answer the question. "
                        "Reply with JSON: {\"sufficient\": true, \"reason\": \"ok\"}."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def content_overlap(question: str, chunks: list[RetrievedChunk]) -> float:
    query_terms = _content_terms(question)
    if not query_terms:
        return 1.0
    passage = " ".join(f"{chunk.title} {chunk.section} {chunk.text}" for chunk in chunks)
    hit = _content_terms(passage)
    return len(query_terms & hit) / len(query_terms)


def check_evidence(
    question: str,
    chunks: list[RetrievedChunk],
    checker: EvidenceChecker | None = None,
) -> EvidenceVerdict:
    return (checker or get_evidence_checker()).check(question, chunks)


def get_evidence_checker(provider: str | None = None) -> EvidenceChecker:
    name = (provider or EVIDENCE_CHECKER).lower()
    if name == "heuristic":
        return HeuristicEvidenceChecker()
    if name == "openai":
        return OpenAIEvidenceChecker()
    raise ValueError(f"Unknown evidence checker: {name}")


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOP and len(token) > 2
    }
