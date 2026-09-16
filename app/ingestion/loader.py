from __future__ import annotations

import json
from html import unescape
from pathlib import Path

from app.config import ARTICLES_PATH, PROVIDERS_PATH, REVIEWS_PATH
from app.models.schemas import ContentType, RawPost

_KNOWN_PROVIDERS = ("Nextiva", "RingCentral")


def load_raw_posts(content_type: ContentType | None = None) -> list[RawPost]:
    posts: list[RawPost] = []
    if content_type in (None, "article"):
        posts.extend(_load_articles())
    if content_type in (None, "review"):
        posts.extend(_load_reviews())
    if content_type in (None, "provider"):
        posts.extend(_load_providers())
    return posts


def load_all_raw_posts() -> list[RawPost]:
    return load_raw_posts()


def get_raw_post(post_id: str | int, content_type: ContentType | None = None) -> RawPost:
    target = str(post_id)
    matches = [post for post in load_raw_posts(content_type) if post.id == target]
    if not matches:
        scope = content_type or "any type"
        raise KeyError(f"No fixture post with id={target!r} ({scope})")
    return matches[0]


def _load_json(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def _clean_url(value: str | None) -> str | None:
    if not value:
        return None
    return unescape(value)


def _infer_provider(*candidates: str | None) -> str | None:
    blob = " ".join(part for part in candidates if part)
    for name in _KNOWN_PROVIDERS:
        if name.lower() in blob.lower():
            return name
    return None


def _load_articles() -> list[RawPost]:
    posts = []
    for row in _load_json(ARTICLES_PATH):
        title = row.get("title") or ""
        posts.append(
            RawPost(
                id=str(row["id"]),
                title=title,
                slug=row.get("slug") or "",
                html=row.get("content") or "",
                content_type="article",
                provider=_infer_provider(title, row.get("slug")),
                url=_clean_url(row.get("guid")),
                published_at=row.get("post_date"),
                updated_at=row.get("post_modified"),
            )
        )
    return posts


def _load_reviews() -> list[RawPost]:
    posts = []
    for row in _load_json(REVIEWS_PATH):
        html = row.get("review") or row.get("content") or ""
        posts.append(
            RawPost(
                id=str(row["id"]),
                title=row.get("title") or "",
                slug=row.get("slug") or "",
                html=html,
                content_type="review",
                provider=row.get("provider"),
                url=_clean_url(row.get("guid")),
                published_at=row.get("post_date"),
                updated_at=row.get("post_modified"),
            )
        )
    return posts


def _load_providers() -> list[RawPost]:
    posts = []
    for row in _load_json(PROVIDERS_PATH):
        name = row.get("name") or row.get("title") or ""
        posts.append(
            RawPost(
                id=str(row["id"]),
                title=name,
                slug=row.get("slug") or "",
                html=row.get("content") or "",
                content_type="provider",
                provider=name or None,
                url=_clean_url(row.get("guid")),
                published_at=row.get("post_date"),
                updated_at=row.get("post_modified"),
            )
        )
    return posts
