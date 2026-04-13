from typing import Optional

from kg_gen import KGGen

import config
from extraction.ner_coverage import _get_nlp
from extraction.prompts import RESPONSE_CONTEXT, SOURCE_CONTEXT

_client: Optional[KGGen] = None


def _get_client() -> KGGen:
    global _client
    if _client is None:
        api_key = config.kggen_api_key()
        if not api_key:
            raise RuntimeError(
                f"No API key available for KGGEN_MODEL={config.KGGEN_MODEL!r}. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env."
            )
        _client = KGGen(
            model=config.KGGEN_MODEL,
            temperature=config.KGGEN_TEMPERATURE,
            api_key=api_key,
        )
    return _client


def _ner_entity_set(text: str) -> set[str]:
    nlp = _get_nlp()
    doc = nlp(text)
    entities: set[str] = set()
    for ent in doc.ents:
        entities.add(ent.text.lower())
        for token in ent.text.lower().split():
            entities.add(token)
    return entities


def _entity_in_ner(entity: str, ner_entities: set[str]) -> bool:
    """Permissive case-insensitive match: exact, token membership, or substring overlap."""
    e = entity.lower().strip()
    if not e:
        return False
    if e in ner_entities:
        return True
    for ner in ner_entities:
        if e == ner or e in ner or ner in e:
            return True
    return False


def _ner_confidence(triple: dict, ner_entities: set[str]) -> str:
    e1_found = _entity_in_ner(triple["entity1"], ner_entities)
    e2_found = _entity_in_ner(triple["entity2"], ner_entities)
    if e1_found and e2_found:
        return "HIGH"
    if e1_found or e2_found:
        return "MEDIUM"
    return "LOW"


def _kggen_to_triples(graph, ner_entities: set[str]) -> list[dict]:
    triples: list[dict] = []
    relations = getattr(graph, "relations", None) or []
    seen: set[tuple[str, str, str]] = set()
    for rel in relations:
        try:
            e1, relation, e2 = rel
        except (TypeError, ValueError):
            continue
        e1, relation, e2 = str(e1).strip(), str(relation).strip(), str(e2).strip()
        if not (e1 and relation and e2):
            continue
        key = (e1.lower(), relation.lower(), e2.lower())
        if key in seen:
            continue
        seen.add(key)
        triples.append(
            {
                "entity1": e1,
                "relation": relation,
                "entity2": e2,
                "confidence": _ner_confidence(
                    {"entity1": e1, "entity2": e2}, ner_entities
                ),
            }
        )
    return triples


def _clusters_to_dict(clusters) -> dict:
    """Normalize kg-gen's cluster structure (Dict[str, Set[str]] or similar) to a plain dict."""
    if not clusters:
        return {}
    out: dict[str, list[str]] = {}
    if isinstance(clusters, dict):
        for canonical, members in clusters.items():
            out[str(canonical)] = sorted(str(m) for m in members)
        return out
    try:
        for item in clusters:
            if isinstance(item, dict) and "canonical" in item:
                out[str(item["canonical"])] = sorted(
                    str(m) for m in item.get("members", [])
                )
    except TypeError:
        pass
    return out


def extract_kg(text: str, *, is_source: bool) -> dict:
    """Run kg-gen on text and return a dict with triples + clustering metadata."""
    if not text or not text.strip():
        return {"triples": [], "entity_clusters": {}, "edge_clusters": {}, "entities": [], "edges": []}

    client = _get_client()
    kwargs = dict(
        input_data=text,
        cluster=True,
        context=SOURCE_CONTEXT if is_source else RESPONSE_CONTEXT,
    )
    if is_source:
        kwargs["chunk_size"] = config.KGGEN_CHUNK_SIZE

    graph = client.generate(**kwargs)
    ner_entities = _ner_entity_set(text)
    triples = _kggen_to_triples(graph, ner_entities)

    return {
        "triples": triples,
        "entity_clusters": _clusters_to_dict(getattr(graph, "entity_clusters", None)),
        "edge_clusters": _clusters_to_dict(getattr(graph, "edge_clusters", None)),
        "entities": sorted(str(e) for e in (getattr(graph, "entities", None) or [])),
        "edges": sorted(str(e) for e in (getattr(graph, "edges", None) or [])),
    }


def extract_triples(text: str, *, is_source: bool = True) -> list[dict]:
    """Backwards-compatible convenience wrapper: returns only the triple list."""
    return extract_kg(text, is_source=is_source)["triples"]
