"""Export published GetVoIP posts from local MySQL into root fixture JSON files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import ARTICLES_PATH, PROJECT_ROOT, PROVIDERS_PATH, REVIEWS_PATH

_REVIEW_META = (
    "cons",
    "pros",
    "provider",
    "review",
    "rating_price",
    "rating_total",
    "rating_quality",
    "rating_support",
    "rating_features",
    "rating_reliability",
    "rating_installation",
    "verified_review",
    "would_recommend",
)


def _mysql_xml(sql: str) -> list[dict[str, str]]:
    command = [
        "mysql",
        "-h",
        os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "-P",
        os.environ.get("MYSQL_PORT", "3306"),
        "-u",
        os.environ.get("MYSQL_USER", "root"),
        f"-p{os.environ.get('MYSQL_PASSWORD', '')}",
        os.environ.get("MYSQL_DATABASE", "getvoip"),
        "--default-character-set=utf8mb4",
        "--xml",
        "-e",
        sql,
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace") or "mysql failed")
    return _parse_mysql_xml(completed.stdout)


def _parse_mysql_xml(payload: bytes) -> list[dict[str, str]]:
    if not payload.strip():
        return []
    root = ET.fromstring(payload)
    rows: list[dict[str, str]] = []
    for row in root.findall(".//row"):
        record: dict[str, str] = {}
        for field in row.findall("field"):
            name = field.attrib.get("name")
            if not name:
                continue
            if field.attrib.get("xsi:nil") == "true" or field.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
                record[name] = ""
            else:
                record[name] = field.text or ""
        rows.append(record)
    return rows


def _in_list(ids: list[str]) -> str:
    return ",".join(str(int(post_id)) for post_id in ids)


def _meta_by_post(ids: list[str], keys: tuple[str, ...]) -> dict[str, dict[str, str]]:
    if not ids:
        return {}
    quoted = ",".join(f"'{key}'" for key in keys)
    rows = _mysql_xml(
        "SELECT post_id, meta_key, meta_value FROM gv_postmeta "
        f"WHERE post_id IN ({_in_list(ids)}) AND meta_key IN ({quoted})"
    )
    by_post: dict[str, dict[str, str]] = {}
    for row in rows:
        by_post.setdefault(row["post_id"], {})[row["meta_key"]] = row.get("meta_value") or ""
    return by_post


def _provider_names(ids: list[str]) -> dict[str, str]:
    unique = [post_id for post_id in dict.fromkeys(ids) if post_id and post_id.isdigit()]
    if not unique:
        return {}
    rows = _mysql_xml(
        f"SELECT ID, post_title FROM gv_posts WHERE ID IN ({_in_list(unique)})"
    )
    return {row["ID"]: row.get("post_title") or "" for row in rows}


def export_articles(limit: int = 1000) -> list[dict]:
    rows = _mysql_xml(
        "SELECT ID, guid, post_name, post_title, post_content, post_excerpt, "
        "post_date, post_type, post_status, post_modified "
        "FROM gv_posts "
        "WHERE post_type IN ('blog_post','news_post','library_post') AND post_status='publish' "
        "ORDER BY post_date DESC, ID DESC "
        f"LIMIT {int(limit)}"
    )
    return [
        {
            "id": int(row["ID"]),
            "guid": row.get("guid") or "",
            "slug": row.get("post_name") or "",
            "title": row.get("post_title") or "",
            "content": row.get("post_content") or "",
            "excerpt": row.get("post_excerpt") or "",
            "post_date": row.get("post_date") or "",
            "post_type": row.get("post_type") or "blog_post",
            "post_status": row.get("post_status") or "publish",
            "post_modified": row.get("post_modified") or "",
        }
        for row in rows
    ]


def export_providers(limit: int = 300) -> list[dict]:
    rows = _mysql_xml(
        "SELECT ID, guid, post_name, post_title, post_content, "
        "post_date, post_type, post_status, post_modified "
        "FROM gv_posts "
        "WHERE post_type='provider' AND post_status='publish' "
        "ORDER BY post_modified DESC, post_date DESC, ID DESC "
        f"LIMIT {int(limit)}"
    )
    meta = _meta_by_post([row["ID"] for row in rows], ("meta_description",))
    return [
        {
            "id": int(row["ID"]),
            "guid": row.get("guid") or "",
            "name": row.get("post_title") or "",
            "slug": row.get("post_name") or "",
            "content": row.get("post_content") or "",
            "post_date": row.get("post_date") or "",
            "post_type": row.get("post_type") or "provider",
            "post_status": row.get("post_status") or "publish",
            "post_modified": row.get("post_modified") or "",
            "meta_description": meta.get(row["ID"], {}).get("meta_description") or "",
        }
        for row in rows
    ]


def export_reviews(limit: int = 3000) -> list[dict]:
    rows = _mysql_xml(
        "SELECT ID, guid, post_name, post_title, post_content, "
        "post_date, post_type, post_status, post_modified "
        "FROM gv_posts "
        "WHERE post_type='user-reviews' AND post_status='publish' "
        "ORDER BY post_date DESC, ID DESC "
        f"LIMIT {int(limit)}"
    )
    ids = [row["ID"] for row in rows]
    meta = _meta_by_post(ids, _REVIEW_META)
    provider_ids = [meta.get(post_id, {}).get("provider") or "" for post_id in ids]
    names = _provider_names(provider_ids)
    exported: list[dict] = []
    for row in rows:
        fields = meta.get(row["ID"], {})
        provider_id = fields.get("provider") or ""
        exported.append(
            {
                "id": int(row["ID"]),
                "cons": fields.get("cons") or "",
                "guid": row.get("guid") or "",
                "pros": fields.get("pros") or "",
                "slug": row.get("post_name") or "",
                "title": row.get("post_title") or "",
                "review": fields.get("review") or "",
                "content": row.get("post_content") or "",
                "provider": names.get(provider_id) or "",
                "post_date": row.get("post_date") or "",
                "post_type": row.get("post_type") or "user-reviews",
                "post_status": row.get("post_status") or "publish",
                "provider_id": int(provider_id) if provider_id.isdigit() else None,
                "rating_price": fields.get("rating_price") or "",
                "rating_total": fields.get("rating_total") or "",
                "post_modified": row.get("post_modified") or "",
                "rating_quality": fields.get("rating_quality") or "",
                "rating_support": fields.get("rating_support") or "",
                "rating_features": fields.get("rating_features") or "",
                "verified_review": fields.get("verified_review") or "",
                "would_recommend": fields.get("would_recommend") or "",
                "rating_reliability": fields.get("rating_reliability") or "",
                "rating_installation": fields.get("rating_installation") or "",
            }
        )
    return exported


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    print(f"Exporting fixtures from {os.environ.get('MYSQL_HOST')}:{os.environ.get('MYSQL_PORT')}", flush=True)
    articles = export_articles(1000)
    print(f"articles {len(articles)} -> {ARTICLES_PATH}", flush=True)
    _write(ARTICLES_PATH, articles)
    reviews = export_reviews(3000)
    print(f"reviews {len(reviews)} -> {REVIEWS_PATH}", flush=True)
    _write(REVIEWS_PATH, reviews)
    providers = export_providers(300)
    print(f"providers {len(providers)} -> {PROVIDERS_PATH}", flush=True)
    _write(PROVIDERS_PATH, providers)
    print(f"done ({PROJECT_ROOT})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
