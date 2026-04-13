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


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_pipeline(source_text: str, response_text: str) -> dict:
    # Stage 2: extract KGs from both sides via kg-gen (chunking + clustering)
    source_kg = extract_kg(source_text, is_source=True)
    response_kg = extract_kg(response_text, is_source=False)
    source_triples = source_kg["triples"]
    response_triples = response_kg["triples"]

    # Stage 2 (continued): NER coverage on the source side
    coverage = check_coverage(source_text, source_triples)

    # Aggregate source KG confidence, used by classifier for no-match decisions
    source_kg_confidence = aggregate_source_confidence(source_triples)
    for t in response_triples:
        t["_source_kg_confidence"] = source_kg_confidence

    # Stage 3: semantic matching
    matches = match_triples(response_triples, source_triples)

    # Stages 4 + 5: classify, with NLI comparison for matched triples and
    # NLI fallback for uncertain verdicts
    verdicts: list[dict] = []
    for match in matches:
        response_triple = match["response_triple"]
        best_source = match["best_source_triple"]
        similarity = match["similarity"]

        nli_entailment = None
        if best_source is not None and similarity >= config.SIMILARITY_THRESHOLD:
            nli_entailment = run_nli(
                triple_to_sentence(best_source),
                triple_to_sentence(response_triple),
            )

        result = classify(match, nli_entailment=nli_entailment)

        if result["verdict"] == UNCERTAIN:
            fallback = resolve_uncertain(response_triple, source_text, source_triples)
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
    }
    faithfulness = counts[GROUNDED] / total if total else 1.0

    return {
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
        "--output",
        help="Optional path to write JSON report. Defaults to stdout.",
    )
    args = parser.parse_args()

    report = run_pipeline(_read(args.source), _read(args.response))
    serialized = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
