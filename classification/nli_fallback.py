from typing import Optional

from transformers import AutoModelForSequenceClassification

import config
from classification.classifier import CONTRADICTED, FABRICATED, GROUNDED
from matching.embedder import triple_to_sentence
from matching.entity_aligner import align_entities

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(
            config.NLI_MODEL, trust_remote_code=True
        )
        _model.eval()
    return _model


def run_nli(premise: str, hypothesis: str) -> float:
    """Return HHEM faithfulness/entailment score in [0, 1] for (premise, hypothesis)."""
    model = _get_model()
    scores = model.predict([(premise, hypothesis)])
    score = scores[0] if hasattr(scores, "__iter__") else scores
    try:
        return float(score.item())
    except AttributeError:
        return float(score)


def resolve_uncertain(
    response_triple: dict, raw_source: str, source_triples: list[dict]
) -> dict:
    """Stage 5: resolve an 'uncertain' verdict by falling back to NLI on raw source."""
    hypothesis = triple_to_sentence(response_triple)
    score = run_nli(raw_source, hypothesis)

    if score >= config.NLI_ENTAILMENT_THRESHOLD:
        return {
            "verdict": GROUNDED,
            "reason": f"NLI fallback entailed triple against raw source (nli={score:.3f})",
            "nli_score": score,
        }

    matched_entities = align_entities(response_triple, source_triples)
    if matched_entities:
        return {
            "verdict": CONTRADICTED,
            "reason": f"NLI disagreed (nli={score:.3f}); entities {matched_entities} present in source KG",
            "nli_score": score,
        }

    return {
        "verdict": FABRICATED,
        "reason": f"NLI disagreed (nli={score:.3f}) and no triple entities appear in source KG",
        "nli_score": score,
    }
