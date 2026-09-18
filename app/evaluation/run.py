from __future__ import annotations

import argparse
import logging
import sys

from app.evaluation.dataset import load_eval_cases
from app.evaluation.experiments import (
    COMPARE_STRATEGIES,
    chunker_retrievers,
    run_comparison,
    run_experiment,
    strategy_retrievers,
    write_comparison_report,
    write_experiment_report,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Run golden-set eval. Embedding model is held fixed (OpenAI); Bedrock bake-off is postponed."
    )
    parser.add_argument("--limit", type=int, help="Score only the first N cases")
    parser.add_argument("--id", dest="case_id", help="Score a single eval case id")
    parser.add_argument(
        "--strategy",
        choices=COMPARE_STRATEGIES,
        help="Single retrieval strategy (default: dense)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare dense, sparse, hybrid, and rerank",
    )
    parser.add_argument(
        "--compare-chunkers",
        action="store_true",
        help="Compare dense retrieval on each chunker collection",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval only; skip LLM generation",
    )
    args = parser.parse_args(argv)

    if args.compare and args.compare_chunkers:
        parser.error("Use either --compare or --compare-chunkers")

    cases = load_eval_cases()
    if args.case_id:
        cases = [case for case in cases if case.id == args.case_id]
        if not cases:
            print(f"No eval case with id={args.case_id!r}", file=sys.stderr)
            return 1
    elif args.limit is not None:
        cases = cases[: args.limit]

    generate = not args.retrieval_only
    logging.getLogger("app.evaluation").info(
        "Eval starting cases=%s generate=%s compare=%s compare_chunkers=%s",
        len(cases),
        generate,
        args.compare,
        args.compare_chunkers,
    )
    if args.compare or args.compare_chunkers:
        retrievers = chunker_retrievers() if args.compare_chunkers else strategy_retrievers()
        report = run_comparison(cases, retrievers=retrievers, generate=generate)
        path = write_comparison_report(report)
        print(f"{path}  embedder={report.embedder}  generate={report.generate}")
        _print_comparison(report)
        return 0

    retriever = None
    if args.strategy:
        from app.retrieval import retriever_for

        retriever = retriever_for(args.strategy)
    report = run_experiment(cases, retriever=retriever, generate=generate)
    path = write_experiment_report(report)
    print(
        f"{path}  cases={report.case_count}  "
        f"recall={report.retrieval['recall_at_k']:.3f}  "
        f"mrr={report.retrieval['mrr']:.3f}  "
        f"errors={report.error_rate:.3f}  "
        f"cost_usd={report.engineering['cost_usd_total']:.6f}"
    )
    return 0


def _print_comparison(report) -> None:
    print(f"{'variant':<18} {'recall':>7} {'mrr':>7} {'ndcg':>7} {'ms':>8} {'cost':>10}")
    for name, variant in report.variants.items():
        print(
            f"{name:<18} "
            f"{variant.retrieval['recall_at_k']:7.3f} "
            f"{variant.retrieval['mrr']:7.3f} "
            f"{variant.retrieval['ndcg_at_k']:7.3f} "
            f"{variant.engineering['end_to_end_ms']:8.1f} "
            f"{variant.engineering['cost_usd_total']:10.6f}"
        )


if __name__ == "__main__":
    sys.exit(main())
