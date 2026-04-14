"""Print a full per-stage verdict trace for one benchmark row.

    python -m evaluation.inspect --row-id summeval-0-0

Extraction and NLI are pulled from the disk cache when available, so inspecting
rows that were already seen during an eval run is free.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import config
import main as pipeline_main
from evaluation.benchmarks import load_benchmark
from evaluation.cache import cached_extract_kg, cached_resolve_uncertain, cached_run_nli


def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width, subsequent_indent="    "))


def _triple_str(t: dict | None) -> str:
    if t is None:
        return "∅"
    return f"({t['entity1']}) -[{t['relation']}]-> ({t['entity2']}) [{t.get('confidence', '?')}]"


def inspect(row: dict, ablation: str = "main") -> None:
    print("=" * 110)
    print(f"{row['id']}   benchmark={row['benchmark']}   label={row['label']}   raw_score={row['raw_score']:.3f}")
    print("=" * 110)
    print(f"\nSOURCE ({len(row['source_text'])} chars):")
    print(_wrap(row["source_text"]))
    print(f"\nRESPONSE ({len(row['response_text'])} chars):")
    print(_wrap(row["response_text"]))

    report = pipeline_main.run_pipeline(
        row["source_text"],
        row["response_text"],
        ablation=ablation,
        extract_fn=cached_extract_kg,
        run_nli_fn=cached_run_nli,
        resolve_uncertain_fn=cached_resolve_uncertain,
    )

    print(
        f"\n--- PIPELINE ({ablation}) --- "
        f"faithfulness={report['faithfulness_score']:.3f}  "
        f"source_kg_conf={report['source_kg_confidence']}  "
        f"ner_coverage={report['source_kg_coverage']:.2f}"
    )
    print(f"source triples: {len(report['source_triples'])}   response triples: {report['totals']['response_triples']}")
    print(f"totals: {report['totals']}")

    print("\n--- SOURCE KG ---")
    for t in report["source_triples"]:
        print(f"  {_triple_str(t)}")

    print("\n--- RESPONSE TRIPLES + VERDICTS ---")
    for i, v in enumerate(report["verdicts"], 1):
        print(f"\n[{i}] {_triple_str(v['response_triple'])}")
        print(f"    best source:   {_triple_str(v['best_source_triple'])}")
        sim = v["similarity"]
        nli = v.get("nli_entailment")
        nli_str = f"{nli:.3f}" if nli is not None else "n/a"
        print(f"    similarity={sim:.3f}  nli_entailment={nli_str}")
        print(f"    verdict={v['verdict'].upper()}  ({v['reason']})")
        if v.get("fallback"):
            fb = v["fallback"]
            print(f"    fallback:  nli_score={fb['nli_score']:.3f} -> {fb['verdict']}  ({fb['reason']})")

    # Decision summary
    print("\n--- PREDICTION ---")
    from evaluation.run_eval import PREDICTION_THRESHOLD, binary_prediction

    pred = binary_prediction(report, threshold=PREDICTION_THRESHOLD)
    print(
        f"threshold={PREDICTION_THRESHOLD}  "
        f"faithfulness={report['faithfulness_score']:.3f}  "
        f"→ predicted={'HALLUCINATION' if pred == 1 else 'CONSISTENT'}  "
        f"(label={'HALLUCINATION' if row['label'] == 1 else 'CONSISTENT'})"
    )
    bucket = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(row["label"], pred)]
    print(f"bucket: {bucket}")
    print(f"config: SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD}  NLI_ENTAILMENT_THRESHOLD={config.NLI_ENTAILMENT_THRESHOLD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-id", required=True, help="Row id, e.g. summeval-0-0")
    parser.add_argument("--ablation", default="main", choices=sorted(pipeline_main.ABLATIONS))
    args = parser.parse_args()

    benchmark = args.row_id.rsplit("-", 2 if args.row_id.startswith("summeval") else 1)[0]
    rows = load_benchmark(benchmark)
    match = next((r for r in rows if r["id"] == args.row_id), None)
    if match is None:
        print(f"No row with id {args.row_id!r} in benchmark {benchmark!r}", file=sys.stderr)
        return 1
    inspect(match, ablation=args.ablation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
