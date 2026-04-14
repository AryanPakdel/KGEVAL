"""Disk cache for expensive pipeline operations.

Keyed by SHA-256 of the relevant inputs (text + model id + side). Stored as
one JSON file per entry under `data/cache/{kind}/`. Safe across interrupts —
writes are atomic (tmp file + rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import config
from classification.nli_fallback import run_nli as _run_nli
from extraction.kg_extractor import extract_kg as _extract_kg

CACHE_DIR = Path("data/cache")


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def _read(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _write_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def cached_extract_kg(text: str, *, is_source: bool) -> dict:
    key = _digest("kg", config.KGGEN_MODEL, "source" if is_source else "response", text)
    path = CACHE_DIR / "kg" / f"{key}.json"
    hit = _read(path)
    if hit is not None:
        return hit
    result = _extract_kg(text, is_source=is_source)
    _write_atomic(path, result)
    return result


def cached_run_nli(premise: str, hypothesis: str) -> float:
    key = _digest("nli", config.NLI_MODEL, premise, hypothesis)
    path = CACHE_DIR / "nli" / f"{key}.json"
    hit = _read(path)
    if hit is not None:
        return float(hit["score"])
    score = float(_run_nli(premise, hypothesis))
    _write_atomic(path, {"score": score})
    return score


def cached_resolve_uncertain(response_triple: dict, raw_source: str, source_triples: list[dict]) -> dict:
    """Cached fallback that reuses cached_run_nli under the hood."""
    from classification.classifier import CONTRADICTED, FABRICATED, GROUNDED
    from matching.embedder import triple_to_sentence
    from matching.entity_aligner import align_entities

    hypothesis = triple_to_sentence(response_triple)
    score = cached_run_nli(raw_source, hypothesis)

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
