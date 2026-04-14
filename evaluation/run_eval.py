"""End-to-end benchmark evaluation harness.

Usage:
    python -m evaluation.run_eval --benchmark summeval --limit 50
    python -m evaluation.run_eval --benchmark all --limit 50 --ablation all
    python -m evaluation.run_eval --benchmark qags_c --ablation no_fallback

For each (benchmark, ablation) pair it runs the full pipeline over `--limit`
examples, aggregates per-triple verdicts into a binary prediction, and reports
balanced accuracy + P/R/F1 against the benchmark's human labels.

Extraction and NLI calls are cached to `data/cache/` so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import main as pipeline_main
from evaluation.benchmarks import available_benchmarks, load_benchmark
from evaluation.cache import cached_extract_kg, cached_resolve_uncertain, cached_run_nli
from evaluation.metrics import compute_metrics, format_metrics_table

PREDICTION_THRESHOLD = 0.3  # consistent iff faithfulness >= 0.3 (tuned; GraphEval-style 1.0 is unreachable on real data)


def binary_prediction(report: dict, threshold: float = PREDICTION_THRESHOLD) -> int:
    """1 = hallucination (faithfulness below threshold), 0 = consistent."""
    return 0 if report["faithfulness_score"] >= threshold else 1


def evaluate(
    benchmark: str,
    ablation: str,
    limit: Optional[int] = None,
    threshold: float = PREDICTION_THRESHOLD,
    progress: bool = True,
) -> dict:
    rows = load_benchmark(benchmark, limit=limit)
    y_true: list[int] = []
    y_pred: list[int] = []
    per_row: list[dict] = []
    errors: list[dict] = []

    iterator = tqdm(rows, desc=f"{benchmark}/{ablation}", disable=not progress)
    start = time.time()
    for row in iterator:
        try:
            report = pipeline_main.run_pipeline(
                row["source_text"],
                row["response_text"],
                ablation=ablation,
                extract_fn=cached_extract_kg,
                run_nli_fn=cached_run_nli,
                resolve_uncertain_fn=cached_resolve_uncertain,
            )
            pred = binary_prediction(report, threshold=threshold)
            y_true.append(row["label"])
            y_pred.append(pred)
            per_row.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": pred,
                    "faithfulness_score": report["faithfulness_score"],
                    "totals": report.get("totals"),
                }
            )
        except Exception as e:
            errors.append({"id": row["id"], "error": repr(e), "trace": traceback.format_exc()})
    elapsed = time.time() - start

    metrics = compute_metrics(y_true, y_pred)
    metrics["elapsed_seconds"] = elapsed
    metrics["errors"] = len(errors)
    return {
        "benchmark": benchmark,
        "ablation": ablation,
        "threshold": threshold,
        "metrics": metrics,
        "per_row": per_row,
        "errors": errors,
    }


def _expand(names: list[str], universe) -> list[str]:
    return sorted(universe) if names == ["all"] else names


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=["summeval"],
        help="One or more of summeval/qags_c/qags_x or 'all'.",
    )
    parser.add_argument(
        "--ablation",
        nargs="+",
        default=["main"],
        help="One or more of main/no_fallback/no_graph/no_confidence or 'all'.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Examples per benchmark (default 50).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PREDICTION_THRESHOLD,
        help="Faithfulness threshold for predicting 'consistent'. Default 1.0 (GraphEval-style).",
    )
    parser.add_argument("--output", default="eval_results.json", help="Write full results JSON here.")
    args = parser.parse_args()

    benchmarks = _expand(args.benchmark, available_benchmarks())
    ablations = _expand(args.ablation, list(pipeline_main.ABLATIONS))

    all_results: list[dict] = []
    summary: dict[str, dict] = {}

    for ablation in ablations:
        for benchmark in benchmarks:
            print(f"\n=== {benchmark} | ablation={ablation} | limit={args.limit} ===")
            r = evaluate(benchmark, ablation, limit=args.limit, threshold=args.threshold)
            all_results.append(r)
            summary[f"{benchmark}/{ablation}"] = r["metrics"]
            m = r["metrics"]
            if m["balanced_accuracy"] is None:
                print("  no examples evaluated")
            else:
                print(
                    f"  n={m['n']} | bal_acc={m['balanced_accuracy']*100:.2f} | "
                    f"f1={m['f1']*100:.2f} | errors={m['errors']} | "
                    f"elapsed={m['elapsed_seconds']:.1f}s"
                )

    print("\n" + format_metrics_table(summary))

    out_path = Path(args.output)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote detailed results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
