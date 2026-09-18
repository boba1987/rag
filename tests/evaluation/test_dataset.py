import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import CHUNKED_DIR
from app.evaluation.dataset import categories_present, load_eval_cases
from app.models.schemas import EvalCase, EvalCategory

_PLANNED_CATEGORIES: set[EvalCategory] = {
    "factual",
    "pricing",
    "features",
    "integrations",
    "comparison",
    "recommendation",
    "multi-hop",
    "ambiguous",
    "unanswerable",
}


def _chunked_document_ids() -> set[str]:
    ids: set[str] = set()
    if not CHUNKED_DIR.exists():
        return ids
    for path in CHUNKED_DIR.glob("*.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if rows:
            ids.add(rows[0]["document_id"])
    return ids


def test_loads_the_current_golden_set() -> None:
    cases = load_eval_cases()
    assert len(cases) == 50
    assert cases[0].id == "eval_001"
    assert "NICE CXone pricing" in cases[0].question


def test_categories_are_from_the_plan() -> None:
    present = categories_present(load_eval_cases())
    assert present <= _PLANNED_CATEGORIES
    assert present


def test_ids_and_questions_are_unique() -> None:
    cases = load_eval_cases()
    assert len({case.id for case in cases}) == len(cases)
    assert len({case.question for case in cases}) == len(cases)


def test_expected_documents_exist_in_chunked_corpus() -> None:
    known = _chunked_document_ids()
    assert known, "documents/chunked is empty; run chunking first"
    for case in load_eval_cases():
        missing = set(case.expected_documents) - known
        assert not missing, f"{case.id} references unknown documents: {missing}"


def test_ambiguous_and_unanswerable_have_no_expected_documents() -> None:
    for case in load_eval_cases():
        if case.category in {"ambiguous", "unanswerable"}:
            assert case.expected_documents == [], case.id


def test_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        EvalCase(
            id="bad",
            question="What color is the logo?",
            category="style",  # type: ignore[arg-type]
        )


def test_load_eval_cases_reads_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "mini.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "eval_tmp",
                    "question": "Does RingCentral integrate with Salesforce?",
                    "expected_documents": ["8015"],
                    "expected_answer": "Yes.",
                    "category": "factual",
                }
            ]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(path)
    assert len(cases) == 1
    assert cases[0].id == "eval_tmp"
