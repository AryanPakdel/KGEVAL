from typing import Optional

import config
from classification.confidence import is_high_confidence

GROUNDED = "grounded"
CONTRADICTED = "contradicted"
FABRICATED = "fabricated"
UNCERTAIN = "uncertain"


def classify(
    match_result: dict,
    nli_entailment: Optional[float] = None,
) -> dict:
    """Stage 4 classification for a single response triple.

    match_result: {response_triple, best_source_triple, similarity}
    nli_entailment: score in [0, 1] comparing response triple to best source
        triple sentence. Required when similarity > SIMILARITY_THRESHOLD,
        ignored otherwise.

    Returns {verdict, reason} where verdict is one of
    grounded / contradicted / fabricated / uncertain.
    """
    similarity = match_result["similarity"]
    best_source = match_result["best_source_triple"]
    response_triple = match_result["response_triple"]

    if best_source is not None and similarity >= config.SIMILARITY_THRESHOLD:
        if nli_entailment is None:
            raise ValueError(
                "NLI entailment score required when similarity is above threshold."
            )
        if nli_entailment >= config.NLI_ENTAILMENT_THRESHOLD:
            return {
                "verdict": GROUNDED,
                "reason": f"matched source triple (sim={similarity:.3f}, nli={nli_entailment:.3f})",
            }
        return {
            "verdict": CONTRADICTED,
            "reason": f"matched source triple but NLI disagrees (sim={similarity:.3f}, nli={nli_entailment:.3f})",
        }

    source_confidence = response_triple.get("_source_kg_confidence", "LOW")
    if is_high_confidence(source_confidence):
        return {
            "verdict": FABRICATED,
            "reason": f"no match (sim={similarity:.3f}); source KG confidence was HIGH",
        }

    return {
        "verdict": UNCERTAIN,
        "reason": f"no match (sim={similarity:.3f}); source KG confidence was {source_confidence} — deferring to NLI",
    }
