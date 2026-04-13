def _collect_source_entities(source_triples: list[dict]) -> set[str]:
    entities: set[str] = set()
    for t in source_triples:
        entities.add(t["entity1"].lower())
        entities.add(t["entity2"].lower())
    return entities


def align_entities(triple: dict, source_triples: list[dict]) -> list[str]:
    """Return the subset of a triple's entities that appear in the source KG.

    Case-insensitive exact match on entity1 / entity2 against any source-side
    entity. Used by the NLI fallback to distinguish Contradicted (entity seen,
    facts wrong) from Fabricated (entity never mentioned).
    """
    source_entities = _collect_source_entities(source_triples)
    matched = []
    if triple["entity1"].lower() in source_entities:
        matched.append(triple["entity1"])
    if triple["entity2"].lower() in source_entities:
        matched.append(triple["entity2"])
    return matched
