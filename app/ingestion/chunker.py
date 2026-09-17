from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import (
    ACTIVE_CHUNKER,
    HARD_MAX_TOKENS,
    OVERLAP_TOKENS,
    TARGET_MAX_TOKENS,
    TARGET_MIN_TOKENS,
    chunked_dir,
)
from app.models.schemas import Chunk, NormalizedDocument, Section

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Whitespace token estimate. Good enough until a tokenizer is added."""
    return len(text.split()) if text.strip() else 0


class Chunker(ABC):
    """Reusable chunking strategy. Structure-aware and fixed-size now; parent-child next."""

    @abstractmethod
    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        raise NotImplementedError


class StructureAwareChunker(Chunker):
    """One section per chunk unless the section exceeds the target token window."""

    def __init__(
        self,
        target_min: int = TARGET_MIN_TOKENS,
        target_max: int = TARGET_MAX_TOKENS,
        hard_max: int = HARD_MAX_TOKENS,
        overlap: int = OVERLAP_TOKENS,
    ) -> None:
        self.target_min = target_min
        self.target_max = target_max
        self.hard_max = hard_max
        self.overlap = overlap

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            if not section.text.strip():
                continue
            parts = self._split_section(section.text)
            for index, part in enumerate(parts, start=1):
                chunks.append(self._to_chunk(document, section, part, index))
        return chunks

    def _split_section(self, text: str) -> list[str]:
        if count_tokens(text) <= self.target_max:
            return [text.strip()]
        return self._window(self._units(text))

    def _units(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs or [text.strip()]:
            if count_tokens(paragraph) <= self.target_max:
                units.append(paragraph)
                continue
            sentences = [part.strip() for part in _SENTENCE_RE.split(paragraph) if part.strip()]
            units.extend(sentences or [paragraph])
        return units

    def _window(self, units: list[str]) -> list[str]:
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = count_tokens(unit)
            if current and current_tokens + unit_tokens > self.target_max:
                windows.append("\n\n".join(current))
                current, current_tokens = self._overlap_seed(current)
            if unit_tokens > self.hard_max:
                if current:
                    windows.append("\n\n".join(current))
                    current, current_tokens = [], 0
                windows.extend(self._force_split(unit))
                continue
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            windows.append("\n\n".join(current))
        return windows or [""]

    def _overlap_seed(self, previous: list[str]) -> tuple[list[str], int]:
        seed: list[str] = []
        tokens = 0
        for unit in reversed(previous):
            unit_tokens = count_tokens(unit)
            if seed and tokens + unit_tokens > self.overlap:
                break
            seed.insert(0, unit)
            tokens += unit_tokens
        return seed, tokens

    def _force_split(self, text: str) -> list[str]:
        words = text.split()
        size = max(self.target_min, 1)
        step = max(size - self.overlap, 1)
        parts = []
        for start in range(0, len(words), step):
            piece = words[start : start + size]
            if piece:
                parts.append(" ".join(piece))
            if start + size >= len(words):
                break
        return parts

    def _to_chunk(
        self,
        document: NormalizedDocument,
        section: Section,
        text: str,
        index: int,
    ) -> Chunk:
        slug = _SLUG_RE.sub("_", section.heading).strip("_").lower()[:40] or "section"
        return Chunk(
            id=f"{document.content_type}_{document.id}_{slug}_{index:02d}",
            document_id=document.id,
            content_type=document.content_type,
            provider=document.provider,
            title=document.title,
            section=section.heading,
            heading_path=list(section.heading_path),
            text=text,
            source_url=document.url,
            updated_at=document.updated_at,
        )


class FixedSizeChunker(Chunker):
    """Token-window baseline. Ignores H2/H3; slides a fixed window over the full body."""

    def __init__(
        self,
        size: int = TARGET_MAX_TOKENS,
        overlap: int = OVERLAP_TOKENS,
    ) -> None:
        self.size = max(size, 1)
        self.overlap = max(overlap, 0)

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        body = "\n\n".join(section.text.strip() for section in document.sections if section.text.strip())
        words = body.split()
        if not words:
            return []
        step = max(self.size - self.overlap, 1)
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(words), step), start=1):
            piece = words[start : start + self.size]
            if not piece:
                continue
            chunks.append(self._to_chunk(document, " ".join(piece), index))
            if start + self.size >= len(words):
                break
        return chunks

    def _to_chunk(self, document: NormalizedDocument, text: str, index: int) -> Chunk:
        heading = document.title or "Document"
        return Chunk(
            id=f"{document.content_type}_{document.id}_fixed_{index:02d}",
            document_id=document.id,
            content_type=document.content_type,
            provider=document.provider,
            title=document.title,
            section=heading,
            heading_path=[heading],
            text=text,
            source_url=document.url,
            updated_at=document.updated_at,
        )


def get_chunker(name: str | None = None) -> Chunker:
    chunker = name or ACTIVE_CHUNKER
    if chunker == "structure_aware":
        return StructureAwareChunker()
    if chunker == "fixed_size":
        return FixedSizeChunker()
    if chunker == "parent_child":
        raise NotImplementedError("ParentChildChunker is Phase 9 commit 2.")
    raise ValueError(f"Unknown chunker: {chunker}")


def write_chunks(source_name: str, chunks: list[Chunk], directory: Path | None = None) -> Path:
    target = directory or chunked_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / Path(source_name).name
    path.write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path

