from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.config import (
    DENSE_TOP_K,
    INDEXED_DIR,
    OPENAI_EMBED_USD_PER_1M,
    OPENAI_INPUT_USD_PER_1M,
    OPENAI_OUTPUT_USD_PER_1M,
)
from app.evaluation.dataset import load_eval_cases
from app.evaluation.generation import GenerationScores, mean_generation_scores, score_generation
from app.evaluation.retrieval import RetrievalScores, mean_retrieval_scores, score_retrieval, unique_document_ids
from app.generation.context import build_context
from app.generation.generator import generate_grounded_answer
from app.generation.prompts import SYSTEM_PROMPT, user_prompt
from app.ingestion.chunker import count_tokens
from app.models.schemas import EvalCase


@dataclass
class EngineeringMetrics:
    end_to_end_ms: float
    retrieval_ms: float
    reranking_ms: float
    llm_ms: float
    embed_tokens: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    cost_usd: float
    error: str | None = None


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str | None
    retrieved_documents: list[str]
    retrieval: dict
    generation: dict
    engineering: dict
    error: str | None = None


@dataclass
class ExperimentReport:
    ran_at: str
    case_count: int
    error_rate: float
    retrieval: dict[str, float]
    generation: dict[str, float]
    engineering: dict[str, float]
    cases: list[CaseResult] = field(default_factory=list)


def estimate_cost_usd(embed_tokens: int, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        embed_tokens * OPENAI_EMBED_USD_PER_1M
        + prompt_tokens * OPENAI_INPUT_USD_PER_1M
        + completion_tokens * OPENAI_OUTPUT_USD_PER_1M
    ) / 1_000_000


def run_eval_case(
    case: EvalCase,
    retriever,
    generator,
    k: int = DENSE_TOP_K,
) -> CaseResult:
    started = perf_counter()
    retrieval_ms = 0.0
    llm_ms = 0.0
    retrieved_ids: list[str] = []
    answer = None
    error = None
    retrieval_scores: RetrievalScores | None = None
    generation_scores: GenerationScores | None = None
    prompt_tokens = 0
    completion_tokens = 0
    embed_tokens = count_tokens(case.question)

    try:
        retrieve_started = perf_counter()
        chunks = retriever.search(case.question, top_k=k)
        retrieval_ms = (perf_counter() - retrieve_started) * 1000
        retrieved_ids = unique_document_ids(chunks)
        context = build_context(chunks)
        prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(user_prompt(case.question, context))
        generate_started = perf_counter()
        grounded = generate_grounded_answer(case.question, chunks, generator=generator)
        llm_ms = (perf_counter() - generate_started) * 1000
        answer = grounded.answer
        completion_tokens = count_tokens(answer)
        retrieval_scores = score_retrieval(retrieved_ids, case.expected_documents, k)
        generation_scores = score_generation(
            answer=answer,
            context=context,
            retrieved_ids=retrieved_ids,
            expected_ids=case.expected_documents,
            expected_answer=case.expected_answer,
            category=case.category,
            k=k,
        )
    except Exception as exc:  # noqa: BLE001 — eval runner must record failures
        error = str(exc)

    engineering = EngineeringMetrics(
        end_to_end_ms=(perf_counter() - started) * 1000,
        retrieval_ms=retrieval_ms,
        reranking_ms=0.0,
        llm_ms=llm_ms,
        embed_tokens=embed_tokens,
        llm_prompt_tokens=prompt_tokens,
        llm_completion_tokens=completion_tokens,
        cost_usd=estimate_cost_usd(embed_tokens, prompt_tokens, completion_tokens),
        error=error,
    )
    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=answer,
        retrieved_documents=retrieved_ids,
        retrieval=_retrieval_dict(retrieval_scores, k),
        generation=_generation_dict(generation_scores),
        engineering=asdict(engineering),
        error=error,
    )


def run_experiment(
    cases: list[EvalCase] | None = None,
    retriever=None,
    generator=None,
    k: int = DENSE_TOP_K,
) -> ExperimentReport:
    from app.generation.generator import get_generator
    from app.retrieval.dense import DenseRetriever

    rows = cases if cases is not None else load_eval_cases()
    searcher = retriever or DenseRetriever()
    backend = generator or get_generator()
    results = [run_eval_case(case, searcher, backend, k=k) for case in rows]
    retrieval_scores = [
        score_retrieval(row.retrieved_documents, case.expected_documents, k)
        for row, case in zip(results, rows, strict=True)
        if row.error is None
    ]
    generation_scores = [
        GenerationScores(
            faithfulness=row.generation["faithfulness"],
            answer_relevance=row.generation["answer_relevance"],
            context_precision=row.generation["context_precision"],
            context_recall=row.generation["context_recall"],
            groundedness=row.generation["groundedness"],
        )
        for row in results
        if row.error is None
    ]
    errors = sum(1 for row in results if row.error)
    engineering_rows = [row.engineering for row in results]
    return ExperimentReport(
        ran_at=datetime.now(timezone.utc).isoformat(),
        case_count=len(results),
        error_rate=errors / len(results) if results else 0.0,
        retrieval=mean_retrieval_scores(retrieval_scores),
        generation=mean_generation_scores(generation_scores),
        engineering=_mean_engineering(engineering_rows, errors, len(results)),
        cases=results,
    )


def write_experiment_report(report: ExperimentReport, path: Path | None = None) -> Path:
    INDEXED_DIR.mkdir(parents=True, exist_ok=True)
    target = path or INDEXED_DIR / "eval-last-run.json"
    payload = {
        "ran_at": report.ran_at,
        "case_count": report.case_count,
        "error_rate": report.error_rate,
        "retrieval": report.retrieval,
        "generation": report.generation,
        "engineering": report.engineering,
        "cases": [asdict(row) for row in report.cases],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _retrieval_dict(scores: RetrievalScores | None, k: int) -> dict:
    if scores is None:
        return {
            "k": k,
            "recall_at_k": None,
            "precision_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "ndcg_at_k": None,
        }
    return asdict(scores)


def _generation_dict(scores: GenerationScores | None) -> dict:
    if scores is None:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": None,
            "groundedness": 0.0,
        }
    return asdict(scores)


def _mean_engineering(rows: list[dict], errors: int, total: int) -> dict[str, float]:
    if not rows:
        return {
            "end_to_end_ms": 0.0,
            "retrieval_ms": 0.0,
            "reranking_ms": 0.0,
            "llm_ms": 0.0,
            "embed_tokens": 0.0,
            "llm_prompt_tokens": 0.0,
            "llm_completion_tokens": 0.0,
            "cost_usd": 0.0,
            "error_rate": 0.0,
        }
    keys = (
        "end_to_end_ms",
        "retrieval_ms",
        "reranking_ms",
        "llm_ms",
        "embed_tokens",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "cost_usd",
    )
    means = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    means["error_rate"] = errors / total if total else 0.0
    means["cost_usd_total"] = sum(row["cost_usd"] for row in rows)
    return means
