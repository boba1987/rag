from app.evaluation.experiments import estimate_cost_usd, run_experiment
from app.models.schemas import EvalCase, RetrievedChunk


class _FakeRetriever:
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                id="provider_8019_core_01",
                document_id="8019",
                content_type="provider",
                provider="Nextiva",
                title="Nextiva",
                section="Nextiva Core ($15-$23/user per month)",
                heading_path=["Nextiva", "Nextiva Core ($15-$23/user per month)"],
                text="Nextiva Core is $15–$23 per user per month.",
                score=0.9,
            )
        ]


class _FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        if "Zoom" in question:
            raise RuntimeError("boom")
        return "Nextiva Core is $15–$23 per user per month."


def test_estimate_cost_scales_with_tokens() -> None:
    assert estimate_cost_usd(0, 0, 0) == 0.0
    assert estimate_cost_usd(1_000_000, 0, 0) > 0


def test_run_experiment_scores_success_and_records_errors() -> None:
    cases = [
        EvalCase(
            id="eval_002",
            question="How much does Nextiva Core cost per user?",
            expected_documents=["8019"],
            expected_answer="Nextiva Core is $15–$23 per user per month.",
            category="pricing",
        ),
        EvalCase(
            id="eval_009",
            question="What is Zoom Phone's 2020 annual revenue?",
            expected_documents=[],
            expected_answer="The indexed GetVoIP corpus does not contain Zoom Phone's 2020 annual revenue.",
            category="unanswerable",
        ),
    ]
    report = run_experiment(cases, retriever=_FakeRetriever(), generator=_FakeGenerator(), k=5)
    assert report.case_count == 2
    assert report.error_rate == 0.5
    success = next(row for row in report.cases if row.id == "eval_002")
    failed = next(row for row in report.cases if row.id == "eval_009")
    assert success.error is None
    assert success.retrieval["recall_at_k"] == 1.0
    assert success.generation["context_recall"] == 1.0
    assert success.engineering["reranking_ms"] == 0.0
    assert failed.error == "boom"
    assert "error_rate" in report.engineering
    assert "cost_usd_total" in report.engineering
