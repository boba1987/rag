from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.cleaner import clean_html

BlockKind = Literal["heading", "paragraph", "list", "table"]
_CONTAINER_TAGS = frozenset({"div", "section", "article", "main", "blockquote"})
_HTML_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ContentBlock:
    kind: BlockKind
    text: str
    level: int | None = None


def parse_blocks(html: str) -> list[ContentBlock]:
    """Parse cleaned HTML (or plain text) into structured content blocks."""
    cleaned = clean_html(html)
    if not cleaned.strip():
        return []
    if not _HTML_RE.search(cleaned):
        return _plain_text_blocks(cleaned)

    soup = BeautifulSoup(cleaned, "lxml")
    root = soup.body or soup
    blocks: list[ContentBlock] = []
    for child in root.children:
        blocks.extend(_blocks_from(child))
    return [block for block in blocks if block.text.strip()]


def _plain_text_blocks(text: str) -> list[ContentBlock]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [ContentBlock("paragraph", _normalize_text(part)) for part in parts if part.strip()]


def _blocks_from(node: Tag | NavigableString) -> list[ContentBlock]:
    if isinstance(node, NavigableString):
        text = _normalize_text(str(node))
        return [ContentBlock("paragraph", text)] if text else []
    if not isinstance(node, Tag) or not node.name:
        return []

    name = node.name
    if name in {"h1", "h2", "h3"}:
        text = _normalize_text(node.get_text(" ", strip=True))
        return [ContentBlock("heading", text, level=int(name[1]))] if text else []
    if name == "p":
        text = _normalize_text(node.get_text(" ", strip=True))
        return [ContentBlock("paragraph", text)] if text else []
    if name in {"ul", "ol"}:
        return _list_block(node)
    if name == "table":
        text = _table_to_text(node)
        return [ContentBlock("table", text)] if text else []
    if name in _CONTAINER_TAGS:
        blocks: list[ContentBlock] = []
        for child in node.children:
            blocks.extend(_blocks_from(child))
        return blocks

    text = _normalize_text(node.get_text(" ", strip=True))
    return [ContentBlock("paragraph", text)] if text else []


def _list_block(node: Tag) -> list[ContentBlock]:
    items = [
        _normalize_text(item.get_text(" ", strip=True))
        for item in node.find_all("li", recursive=False)
    ]
    items = [item for item in items if item]
    if not items:
        return []
    if node.name == "ol":
        lines = [f"{index}. {item}" for index, item in enumerate(items, start=1)]
    else:
        lines = [f"- {item}" for item in items]
    return [ContentBlock("list", "\n".join(lines))]


def _table_to_text(table: Tag) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            _normalize_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.replace("\xa0", " ")).strip()
