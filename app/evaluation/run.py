from __future__ import annotations

import argparse
import sys

from app.evaluation.dataset import load_eval_cases
from app.evaluation.experiments import run_experiment, write_experiment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set eval against the dense RAG baseline.")
    parser.add_argument("--limit", type=int, help="Score only the first N cases")
    parser.add_argument("--id", dest="case_id", help="Score a single eval case id")
    args = parser.parse_args(argv)

    cases = load_eval_cases()
    if args.case_id:
        cases = [case for case in cases if case.id == args.case_id]
        if not cases:
            print(f"No eval case with id={args.case_id!r}", file=sys.stderr)
            return 1
    elif args.limit is not None:
        cases = cases[: args.limit]

    report = run_experiment(cases)
    path = write_experiment_report(report)
    print(
        f"{path}  cases={report.case_count}  "
        f"recall={report.retrieval['recall_at_k']:.3f}  "
        f"mrr={report.retrieval['mrr']:.3f}  "
        f"errors={report.error_rate:.3f}  "
        f"cost_usd={report.engineering['cost_usd_total']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
