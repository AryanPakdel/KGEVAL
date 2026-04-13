import config

CONFIDENCE_SCORES = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}


def confidence_to_score(label: str) -> float:
    return CONFIDENCE_SCORES.get(str(label).upper(), 0.5)


def is_high_confidence(label: str) -> bool:
    return str(label).upper() in config.HIGH_CONFIDENCE_LABELS


def aggregate_source_confidence(triples: list[dict]) -> str:
    """Return an overall HIGH/MEDIUM/LOW confidence label for a KG.

    Uses the mean of per-triple confidence scores:
        >= 0.75 → HIGH, >= 0.25 → MEDIUM, else LOW.
    """
    if not triples:
        return "LOW"
    scores = [confidence_to_score(t.get("confidence", "MEDIUM")) for t in triples]
    mean = sum(scores) / len(scores)
    if mean >= 0.75:
        return "HIGH"
    if mean >= 0.25:
        return "MEDIUM"
    return "LOW"
