import json

from app.evaluation.experiments import (
    COMPARE_STRATEGIES,
    category_retrieval,
    run_comparison,
    run_experiment,
    write_comparison_report,
)
from app.models.schemas import EvalCase, RetrievedChunk


class _DocRetriever:
    def __init__(self, document_id: str) -> None:
        self._document_id = document_id

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                id=f"{self._document_id}_pricing",
                document_id=self._document_id,
                content_type="provider",
                provider="Nextiva" if self._document_id == "8019" else "RingCentral",
                title="Provider",
                section="Pricing",
                heading_path=["Provider", "Pricing"],
                text="Core is $15 per user.",
                score=0.9,
            )
        ]


class _SilentGenerator:
    def generate(self, question: str, context: str) -> str:
        raise AssertionError("retrieval-only should not generate")


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="eval_002",
            question="How much does Nextiva Core cost per user?",
            expected_documents=["8019"],
            expected_answer="Nextiva Core is $15–$23 per user per month.",
            category="pricing",
        ),
        EvalCase(
            id="eval_hop",
            question="Compare Nextiva and RingCentral.",
            expected_documents=["8019", "8015"],
            category="comparison",
        ),
    ]


def test_compare_strategies_are_the_live_retrieval_paths() -> None:
    assert COMPARE_STRATEGIES == ("dense", "sparse", "hybrid", "rerank", "raptor")


def test_run_experiment_retrieval_only_skips_generator() -> None:
    report = run_experiment(
        _cases()[:1],
        retriever=_DocRetriever("8019"),
        generator=_SilentGenerator(),
        k=5,
        generate=False,
    )
    assert report.cases[0].error is None
    assert report.cases[0].answer is None
    assert report.retrieval["recall_at_k"] == 1.0
    assert report.generation["faithfulness"] == 0.0


def test_run_comparison_ranks_variants(tmp_path) -> None:
    report = run_comparison(
        _cases(),
        retrievers={
            "dense": _DocRetriever("8019"),
            "raptor": _DocRetriever("9999"),
        },
        generator=_SilentGenerator(),
        generate=False,
    )
    assert report.embedder
    assert report.generate is False
    assert report.variants["dense"].retrieval["recall_at_k"] > report.variants["raptor"].retrieval["recall_at_k"]
    by_category = category_retrieval(report.variants["dense"].cases, report.cases, k=5)
    assert by_category["pricing"]["recall_at_k"] == 1.0
    assert by_category["comparison"]["recall_at_k"] == 0.5

    path = write_comparison_report(report, path=tmp_path / "eval-compare.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "no Bedrock access" in payload["note"]
    assert set(payload["variants"]) == {"dense", "raptor"}
    assert payload["variants"]["dense"]["by_category"]["pricing"]["recall_at_k"] == 1.0


def test_compare_cli_retrieval_only(tmp_path, monkeypatch) -> None:
    from app.evaluation.experiments import ComparisonReport
    from app.evaluation.run import main

    called: dict = {}

    def fake_compare(cases, retrievers=None, generate=True, **kwargs):
        called["generate"] = generate
        called["count"] = len(cases)
        return ComparisonReport(ran_at="now", embedder="test", generate=generate)

    monkeypatch.setattr("app.evaluation.run.load_eval_cases", _cases)
    monkeypatch.setattr("app.evaluation.run.strategy_retrievers", lambda: {"dense": _DocRetriever("8019")})
    monkeypatch.setattr("app.evaluation.run.run_comparison", fake_compare)
    monkeypatch.setattr("app.evaluation.run.write_comparison_report", lambda report: tmp_path / "eval-compare.json")
    monkeypatch.setattr("app.evaluation.run._print_comparison", lambda report: None)

    assert main(["--compare", "--retrieval-only", "--limit", "1"]) == 0
    assert called["generate"] is False
    assert called["count"] == 1
