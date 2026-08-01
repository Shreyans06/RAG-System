import argparse
import sys

from src.config import get_settings
from src.evaluation.runner import run_eval


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run the SEC filings RAG evaluation pipeline.")
    parser.add_argument(
        "--dataset",
        default=settings.evaluation.get("eval_dataset_path", "eval/datasets/sec_qa_v1.jsonl"),
        help="Path to the eval JSONL dataset",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run first N examples")
    parser.add_argument(
        "--ragas-metrics", default=None,
        help="Subset of ragas metrics"
    )
    args = parser.parse_args()

    ragas_metrics = args.ragas_metrics.split(",") if args.ragas_metrics else None

    report = run_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        ragas_metrics=ragas_metrics,
    )

    print(f"\nRan {len(report.results)} examples\n")
    print(f"{'Metric': <28} {'Value': >8}")
    print("-" * 38)
    for name, value in report.aggregates.items():
        value_str = f"{value:.3f}" if value is not None else "N/A"
        print(f"{name:<28} {value_str: >8}")

    passed = True
    print("\nThreshold checks:")
    for key, threshold in settings.evaluation.items():
        if not key.endswith("_threshold"):
            continue
        metric_name = key[: -len("_threshold")]
        value = report.aggregates.get(metric_name)
        if value is None:
            print(f" {metric_name} : SKIPPED (no score computed)")
            continue
        ok = value >= threshold
        passed = passed and ok
        print(f" {metric_name} : {value:.3f} >= {threshold} -> {'PASS' if ok else 'FAIL'}")

    if not passed:
        print("\nEvaluation FAILED - one or more metrics below threshold.")
        return 1

    print("\nEvaluation PASSED.")
    return 0

if __name__== "__main__":
    sys.exit(main())