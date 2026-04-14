import argparse
import json
import sys
from pathlib import Path

import config
from classification.classifier import (
    CONTRADICTED,
    FABRICATED,
    GROUNDED,
    UNCERTAIN,
    classify,
)
from classification.confidence import aggregate_source_confidence
from classification.nli_fallback import resolve_uncertain, run_nli
from extraction.kg_extractor import extract_kg
from extraction.ner_coverage import check_coverage
from matching.embedder import triple_to_sentence
from matching.triple_matcher import match_triples

ABLATION_MAIN = "main"
ABLATION_NO_FALLBACK = "no_fallback"
ABLATION_NO_GRAPH = "no_graph"
ABLATION_NO_CONFIDENCE = "no_confidence"
ABLATIONS = {ABLATION_MAIN, ABLATION_NO_FALLBACK, ABLATION_NO_GRAPH, ABLATION_NO_CONFIDENCE}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_pipeline(
    source_text: str,
    response_text: str,
    *,
    ablation: str = ABLATION_MAIN,
    extract_fn=extract_kg,
    run_nli_fn=run_nli,
    resolve_uncertain_fn=resolve_uncertain,
) -> dict:
    """Full 6-stage pipeline with optional ablation modes.

    Ablations:
      main           — the proposed system (confidence-gated graph + NLI fallback)
      no_fallback    — skip Stage 5; uncertain stays uncertain (counted as hallucination)
      no_graph       — skip Stages 2-4; run HHEM NLI on raw (source, response) only
      no_confidence  — force source KG confidence = HIGH, so no triple is ever uncertain
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; expected one of {ABLATIONS}")

    if ablation == ABLATION_NO_GRAPH:
        return _run_nli_only(source_text, response_text, run_nli_fn=run_nli_fn)

    source_kg = extract_fn(source_text, is_source=True)
    response_kg = extract_fn(response_text, is_source=False)
    source_triples = source_kg["triples"]
    response_triples = response_kg["triples"]

    coverage = check_coverage(source_text, source_triples)

    if ablation == ABLATION_NO_CONFIDENCE:
        source_kg_confidence = "HIGH"
    else:
        source_kg_confidence = aggregate_source_confidence(source_triples)
    for t in response_triples:
        t["_source_kg_confidence"] = source_kg_confidence

    matches = match_triples(response_triples, source_triples)

    verdicts: list[dict] = []
    for match in matches:
        response_triple = match["response_triple"]
        best_source = match["best_source_triple"]
        similarity = match["similarity"]

        nli_entailment = None
        if best_source is not None and similarity >= config.SIMILARITY_THRESHOLD:
            nli_entailment = run_nli_fn(
                triple_to_sentence(best_source),
                triple_to_sentence(response_triple),
            )

        result = classify(match, nli_entailment=nli_entailment)

        if result["verdict"] == UNCERTAIN and ablation != ABLATION_NO_FALLBACK:
            fallback = resolve_uncertain_fn(response_triple, source_text, source_triples)
            result = {**result, "fallback": fallback, "verdict": fallback["verdict"]}

        verdicts.append(
            {
                "response_triple": _triple_public(response_triple),
                "best_source_triple": _triple_public(best_source) if best_source else None,
                "similarity": similarity,
                "nli_entailment": nli_entailment,
                "verdict": result["verdict"],
                "reason": result["reason"],
                "fallback": result.get("fallback"),
            }
        )

    total = len(verdicts)
    counts = {
        GROUNDED: sum(1 for v in verdicts if v["verdict"] == GROUNDED),
        CONTRADICTED: sum(1 for v in verdicts if v["verdict"] == CONTRADICTED),
        FABRICATED: sum(1 for v in verdicts if v["verdict"] == FABRICATED),
        UNCERTAIN: sum(1 for v in verdicts if v["verdict"] == UNCERTAIN),
    }
    faithfulness = counts[GROUNDED] / total if total else 1.0

    return {
        "ablation": ablation,
        "faithfulness_score": faithfulness,
        "source_kg_coverage": coverage,
        "source_kg_confidence": source_kg_confidence,
        "totals": {"response_triples": total, "source_triples": len(source_triples), **counts},
        "verdicts": verdicts,
        "source_triples": [_triple_public(t) for t in source_triples],
        "clusters": {
            "source_entity_clusters": source_kg["entity_clusters"],
            "source_edge_clusters": source_kg["edge_clusters"],
            "response_entity_clusters": response_kg["entity_clusters"],
            "response_edge_clusters": response_kg["edge_clusters"],
        },
    }


def _run_nli_only(source_text: str, response_text: str, *, run_nli_fn) -> dict:
    """Ablation `no_graph`: just HHEM on (source, response). Baseline-equivalent."""
    score = run_nli_fn(source_text, response_text)
    faithfulness = 1.0 if score >= config.NLI_ENTAILMENT_THRESHOLD else 0.0
    return {
        "ablation": ABLATION_NO_GRAPH,
        "faithfulness_score": faithfulness,
        "nli_score": score,
        "source_kg_coverage": None,
        "source_kg_confidence": None,
        "totals": {"response_triples": 1, "source_triples": 0, GROUNDED: int(faithfulness == 1.0), CONTRADICTED: int(faithfulness == 0.0), FABRICATED: 0, UNCERTAIN: 0},
        "verdicts": [],
        "source_triples": [],
        "clusters": {},
    }


def _triple_public(triple: dict) -> dict:
    return {
        "entity1": triple["entity1"],
        "relation": triple["relation"],
        "entity2": triple["entity2"],
        "confidence": triple.get("confidence", "MEDIUM"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bidirectional KG faithfulness evaluation (Phase 1 CLI)."
    )
    parser.add_argument("--source", required=True, help="Path to source document text file.")
    parser.add_argument("--response", required=True, help="Path to LLM response text file.")
    parser.add_argument(
        "--ablation",
        default=ABLATION_MAIN,
        choices=sorted(ABLATIONS),
        help="Which pipeline variant to run (default: main).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON report. Defaults to stdout.",
    )
    args = parser.parse_args()

    report = run_pipeline(_read(args.source), _read(args.response), ablation=args.ablation)
    serialized = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
