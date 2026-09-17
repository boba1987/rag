from __future__ import annotations

import json
from pathlib import Path

from app.config import QUERY_DIR
from app.query.decomposer import expand_queries
from app.query.extractor import understand_query
from app.query.rewriter import rewrite_query
from app.retrieval.filters import infer_filters

DEFAULT_QUESTIONS = (
    "How much does RingCentral cost?",
    "Does RingCentral integrate with Salesforce?",
    "Compare RingCentral and Nextiva pricing and Salesforce integrations.",
    "What do customers say about Nextiva support?",
    "Who is Five9 better for compared with Dialpad?",
)


def inspect_preprocess(questions: list[str] | None = None, extractor=None, catalog=None) -> list[dict]:
    rows: list[dict] = []
    for question in questions or list(DEFAULT_QUESTIONS):
        extraction = understand_query(question, extractor=extractor, catalog=catalog)
        rows.append(
            {
                "query": question,
                "kind": extraction.kind,
                "extractor": extraction.source,
                "providers": extraction.providers,
                "topics": extraction.topics,
                "rewritten": rewrite_query(question, extraction=extraction, catalog=catalog),
                "queries": expand_queries(question, extraction=extraction, catalog=catalog),
                "filters": infer_filters(question, extraction).model_dump(),
            }
        )
    return rows


def write_preprocess_inspect(
    path: Path | None = None,
    questions: list[str] | None = None,
    extractor=None,
    catalog=None,
) -> Path:
    target = path or QUERY_DIR / "preprocess-inspect.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(inspect_preprocess(questions, extractor=extractor, catalog=catalog), indent=2) + "\n")
    return target
