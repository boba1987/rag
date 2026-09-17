from types import SimpleNamespace

import pytest

from app.models.schemas import RetrievedChunk
from app.query.evidence import (
    HeuristicEvidenceChecker,
    OpenAIEvidenceChecker,
    check_evidence,
    get_evidence_checker,
)


def _chunk(text: str, title: str = "RingCentral") -> RetrievedChunk:
    return RetrievedChunk(
        id="c1",
        document_id="8015",
        content_type="provider",
        provider="RingCentral",
        title=title,
        section="Integrations",
        heading_path=[title, "Integrations"],
        text=text,
        score=0.8,
    )


def test_no_hits_are_insufficient() -> None:
    verdict = check_evidence("Does RingCentral integrate with Salesforce?", [])
    assert verdict.sufficient is False
    assert verdict.reason == "no_hits"


def test_matching_passage_is_sufficient() -> None:
    verdict = check_evidence(
        "Does RingCentral integrate with Salesforce?",
        [_chunk("RingCentral supports Salesforce for automation.")],
    )
    assert verdict.sufficient is True
    assert verdict.reason == "ok"
    assert verdict.overlap >= 0.3


def test_off_topic_passage_is_insufficient() -> None:
    verdict = check_evidence(
        "What is Zoom Phone's 2020 revenue?",
        [_chunk("RingCentral supports Salesforce for automation.")],
    )
    assert verdict.sufficient is False
    assert verdict.reason == "low_overlap"


def test_openai_checker_reads_json() -> None:
    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"sufficient": false, "reason": "unrelated"}')
                    )
                ]
            )

    verdict = OpenAIEvidenceChecker(client=_Client()).check(
        "What is Zoom Phone's 2020 revenue?",
        [_chunk("RingCentral supports Salesforce.")],
    )
    assert verdict.sufficient is False
    assert verdict.reason == "unrelated"


def test_openai_checker_falls_back_when_llm_fails() -> None:
    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("down")

    class _Boom:
        chat = SimpleNamespace(completions=_Completions())

    verdict = OpenAIEvidenceChecker(client=_Boom()).check(
        "Does RingCentral integrate with Salesforce?",
        [_chunk("RingCentral supports Salesforce.")],
    )
    assert verdict.sufficient is True
    assert verdict.reason == "ok"


def test_get_checker_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown evidence checker"):
        get_evidence_checker("nli")


def test_heuristic_is_the_default_checker() -> None:
    assert isinstance(get_evidence_checker(), HeuristicEvidenceChecker)
