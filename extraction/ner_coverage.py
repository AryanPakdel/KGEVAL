from typing import Optional

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
    return _nlp


def check_coverage(raw_text: str, kg_triples: list[dict]) -> float:
    """Return fraction of spaCy NER entities in raw_text that appear in the KG.

    Returns 1.0 when no entities are detected (no evidence of missed coverage).
    """
    nlp = _get_nlp()
    doc = nlp(raw_text)
    ner_entities = set(ent.text.lower() for ent in doc.ents)

    kg_entities: set[str] = set()
    for triple in kg_triples:
        kg_entities.add(triple["entity1"].lower())
        kg_entities.add(triple["entity2"].lower())

    if not ner_entities:
        return 1.0

    covered = ner_entities.intersection(kg_entities)
    return len(covered) / len(ner_entities)
