from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import NORMALIZED_DIR
from app.ingestion.parser import ContentBlock, parse_blocks
from app.models.schemas import NormalizedDocument, RawPost, Section

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_post(post: RawPost) -> NormalizedDocument:
    sections = _sections_from_blocks(post.title, parse_blocks(post.html))
    return NormalizedDocument(
        id=post.id,
        title=post.title,
        content_type=post.content_type,
        provider=post.provider,
        url=post.url,
        published_at=post.published_at,
        updated_at=post.updated_at,
        sections=sections,
    )


def write_normalized_document(post: RawPost, document: NormalizedDocument) -> Path:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    path = NORMALIZED_DIR / _filename(post, document)
    path.write_text(
        json.dumps(document.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sections_from_blocks(title: str, blocks: list[ContentBlock]) -> list[Section]:
    sections: list[Section] = []
    stack: dict[int, str] = {}
    heading = title
    path = [title]
    parts: list[str] = []

    def flush() -> None:
        text = "\n\n".join(part for part in parts if part)
        if text:
            sections.append(Section(heading=heading, heading_path=list(path), text=text))
        parts.clear()

    for block in blocks:
        if block.kind == "heading":
            flush()
            level = block.level or 2
            stack[level] = block.text
            for deeper in [key for key in stack if key > level]:
                del stack[deeper]
            heading = block.text
            tail = [stack[key] for key in sorted(stack)]
            path = [title, *[item for item in tail if item != title]]
            if not path:
                path = [title]
        else:
            parts.append(block.text)
    flush()
    return sections


def _filename(post: RawPost, document: NormalizedDocument) -> str:
    slug = _SLUG_RE.sub("-", post.slug or document.title).strip("-").lower()
    return f"{document.content_type}-{document.id}-{slug or 'doc'}.json"
