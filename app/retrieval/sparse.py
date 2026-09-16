from __future__ import annotations

import json
import math
import re
from collections import Counter

from app.config import CHUNKED_DIR, DENSE_TOP_K
from app.models.schemas import Chunk, RetrievalFilters, RetrievedChunk
from app.retrieval.filters import apply_section_filter, infer_filters, merge_filters

_TOKEN_RE = re.compile(r"[a-z0-9$]+")
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sparse_document(chunk: Chunk) -> str:
    """Index title, section, provider, and body so lexical terms can hit headings."""
    return " ".join(part for part in (chunk.title, chunk.section, chunk.provider or "", chunk.text) if part)


def load_chunk_corpus(directory=CHUNKED_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not directory.exists():
        return chunks
    for path in sorted(directory.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(Chunk.model_validate(row) for row in rows)
    return chunks


class SparseRetriever:
    """In-process BM25 over the chunk corpus. No OpenAI call."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        top_k: int = DENSE_TOP_K,
    ) -> None:
        self._chunks = chunks if chunks is not None else load_chunk_corpus()
        self._top_k = top_k
        tokenized = [tokenize(sparse_document(chunk)) for chunk in self._chunks]
        self._tf = [Counter(tokens) for tokens in tokenized]
        self._dl = [len(tokens) for tokens in tokenized]
        self._avgdl = (sum(self._dl) / len(self._dl)) if self._dl else 0.0
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        n = len(self._chunks)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        infer: bool = False,
    ) -> list[RetrievedChunk]:
        limit = top_k or self._top_k
        applied = merge_filters(filters, infer_filters(query)) if infer else filters
        query_tokens = tokenize(query)
        scored: list[tuple[float, Chunk]] = []
        for chunk, tf, length in zip(self._chunks, self._tf, self._dl, strict=True):
            if not _matches_payload(chunk, applied):
                continue
            score = self._score(query_tokens, tf, length)
            if score <= 0:
                continue
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [
            RetrievedChunk(**chunk.model_dump(), score=score)
            for score, chunk in scored
        ]
        section = applied.section if applied else None
        return apply_section_filter(results, section)[:limit]

    def _score(self, query_tokens: list[str], tf: Counter[str], length: int) -> float:
        if not query_tokens or not self._avgdl:
            return 0.0
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + _K1 * (1 - _B + _B * length / self._avgdl)
            score += idf * (freq * (_K1 + 1)) / denom
        return score


def _matches_payload(chunk: Chunk, filters: RetrievalFilters | None) -> bool:
    if filters is None:
        return True
    if filters.provider and chunk.provider != filters.provider:
        return False
    if filters.content_type and chunk.content_type != filters.content_type:
        return False
    if filters.document_id and chunk.document_id != filters.document_id:
        return False
    return True
