"""Live integration tests for the kg-gen based extractor.

Skipped when no LLM API key is configured. Run with:
    pytest tests/test_extraction.py -s
"""

import pytest

import config
from extraction.kg_extractor import extract_kg

SOURCE = (
    "Albert Einstein was born on 14 March 1879 in Ulm, Germany. "
    "He grew up in Munich where his father Hermann ran an electrical "
    "equipment company. Einstein attended the Luitpold Gymnasium."
)

RESPONSE = (
    "Albert Einstein was born on 14 March 1879 in Munich, Germany. "
    "His father Hermann owned a bookshop. Einstein attended the "
    "Luitpold Gymnasium and later studied at ETH Zurich."
)


pytestmark = pytest.mark.skipif(
    not config.kggen_api_key(),
    reason="No LLM API key configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY).",
)


def _entity_bag(triples: list[dict]) -> set[str]:
    bag: set[str] = set()
    for t in triples:
        bag.add(t["entity1"].lower())
        bag.add(t["entity2"].lower())
    return bag


def _relation_phrases(triples: list[dict]) -> list[str]:
    return [
        f"{t['entity1']} {t['relation']} {t['entity2']}".lower() for t in triples
    ]


def test_source_extraction_produces_reasonable_triples():
    kg = extract_kg(SOURCE, is_source=True)
    triples = kg["triples"]

    assert triples, "kg-gen returned no triples for the source text"

    for t in triples:
        assert t["entity1"] and t["relation"] and t["entity2"]
        assert t["confidence"] in {"HIGH", "MEDIUM", "LOW"}

    entities = _entity_bag(triples)
    assert any("einstein" in e for e in entities), f"Einstein missing: {entities}"
    assert any("ulm" in e for e in entities), f"Ulm missing: {entities}"

    phrases = " | ".join(_relation_phrases(triples))
    assert "luitpold" in phrases.lower(), f"Luitpold relation missing: {phrases}"


def test_response_extraction_produces_reasonable_triples():
    kg = extract_kg(RESPONSE, is_source=False)
    triples = kg["triples"]

    assert triples, "kg-gen returned no triples for the response text"
    entities = _entity_bag(triples)
    assert any("einstein" in e for e in entities)
    assert any("eth" in e or "zurich" in e for e in entities), (
        f"ETH/Zurich missing: {entities}"
    )


def test_clustering_canonicalizes_einstein_references():
    """With cluster=True, 'He' / 'Einstein' / 'Albert Einstein' should all
    surface as a single canonical entity in the extracted triples."""
    kg = extract_kg(SOURCE, is_source=True)

    entities_lower = {e.lower() for e in kg["entities"]}
    einstein_forms = {e for e in entities_lower if "einstein" in e}
    assert len(einstein_forms) == 1, (
        f"Expected exactly one canonical Einstein entity post-clustering; got {einstein_forms}"
    )
    assert einstein_forms == {"albert einstein"}, (
        f"Expected canonical form 'Albert Einstein'; got {einstein_forms}"
    )

    # And the pronoun "He" / "Einstein" alone should never appear as an entity.
    assert "he" not in entities_lower
    assert "einstein" not in entities_lower


def test_edge_clustering_merges_relation_variants():
    """kg-gen's edge clusters should fold variant phrasings of the same relation."""
    kg = extract_kg(SOURCE, is_source=True)
    edge_clusters = kg["edge_clusters"]

    if not edge_clusters:
        pytest.skip("kg-gen returned no edge clusters for this input.")

    merged = [
        (canonical, members)
        for canonical, members in edge_clusters.items()
        if len(members) > 1
    ]
    assert merged, (
        f"Expected at least one edge cluster with multiple members; got {edge_clusters}"
    )


def test_ner_based_confidence_labels_are_assigned():
    kg = extract_kg(SOURCE, is_source=True)
    labels = {t["confidence"] for t in kg["triples"]}
    assert labels, "No confidence labels assigned"
    assert labels.issubset({"HIGH", "MEDIUM", "LOW"})
    assert "HIGH" in labels, (
        "At least one triple should hit HIGH confidence (both entities in NER) "
        f"for this Einstein text; got labels={labels}"
    )
