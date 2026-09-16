from __future__ import annotations

import json
from pathlib import Path

from app.config import GOLDEN_EVAL_PATH
from app.models.schemas import EvalCase, EvalCategory


def load_eval_cases(path: Path | None = None) -> list[EvalCase]:
    target = path or GOLDEN_EVAL_PATH
    records = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {target}")
    return [EvalCase.model_validate(row) for row in records]


def categories_present(cases: list[EvalCase]) -> set[EvalCategory]:
    return {case.category for case in cases}
