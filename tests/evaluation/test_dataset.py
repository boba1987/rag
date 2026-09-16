import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import categories_present, load_eval_cases
from app.models.schemas import EvalCase


def test_loads_starter_golden_set() -> None:
    cases = load_eval_cases()
    assert len(cases) == 9
    assert cases[0].id == "eval_001"
    assert cases[0].question == "Does RingCentral integrate with Salesforce?"
    assert cases[0].expected_documents == ["8015"]
    assert cases[0].category == "factual"


def test_starter_set_covers_every_planned_category() -> None:
    cases = load_eval_cases()
    assert categories_present(cases) == {
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


def test_unanswerable_and_ambiguous_have_no_expected_documents() -> None:
    cases = {case.category: case for case in load_eval_cases()}
    assert cases["ambiguous"].expected_documents == []
    assert cases["unanswerable"].expected_documents == []


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
