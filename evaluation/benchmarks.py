"""Benchmark loaders for SummEval, QAGS-C, QAGS-X.

All three are normalized to a common row format:
    {
        "id":           str,
        "benchmark":    "summeval" | "qags_c" | "qags_x",
        "source_text":  str,   # the original article
        "response_text":str,   # the candidate summary / response
        "label":        0 | 1, # 0 = consistent, 1 = contains hallucination
        "raw_score":    float, # aggregated human consistency score (for debugging)
    }

Label convention follows GraphEval / GraphEval+: positive class = hallucination.
This is what balanced_accuracy is measured against.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

# ---- thresholds for binarizing human consistency scores -----------------------
SUMMEVAL_CONSISTENT_THRESHOLD = 4.0   # mean expert rating (1-5); >=4 → consistent
QAGS_CONSISTENT_THRESHOLD = 0.5       # fraction of "yes" responses; >0.5 per sentence + all sentences → consistent

QAGS_C_URL = "https://raw.githubusercontent.com/W4ngatang/qags/master/data/mturk_cnndm.jsonl"
QAGS_X_URL = "https://raw.githubusercontent.com/W4ngatang/qags/master/data/mturk_xsum.jsonl"

_CACHE_DIR = Path("data/benchmarks")


def _majority_yes(responses: list[dict]) -> bool:
    votes = [1 if str(r.get("response", "")).lower().startswith("y") else 0 for r in responses]
    if not votes:
        return True
    return sum(votes) / len(votes) > QAGS_CONSISTENT_THRESHOLD


def _download(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _load_qags(url: str, benchmark: str, cache_name: str) -> list[dict]:
    raw = _download(url, _CACHE_DIR / cache_name)
    rows: list[dict] = []
    for i, line in enumerate(l for l in raw.splitlines() if l.strip()):
        record = json.loads(line)
        article = record["article"]
        sentences = record["summary_sentences"]
        summary = " ".join(s["sentence"].strip() for s in sentences)
        per_sent_consistent = [_majority_yes(s.get("responses", [])) for s in sentences]
        consistent = all(per_sent_consistent) if per_sent_consistent else True
        raw_score = (
            sum(per_sent_consistent) / len(per_sent_consistent) if per_sent_consistent else 1.0
        )
        rows.append(
            {
                "id": f"{benchmark}-{i}",
                "benchmark": benchmark,
                "source_text": article,
                "response_text": summary,
                "label": 0 if consistent else 1,
                "raw_score": raw_score,
            }
        )
    return rows


def load_qags_c() -> list[dict]:
    return _load_qags(QAGS_C_URL, "qags_c", "qags_c.jsonl")


def load_qags_x() -> list[dict]:
    return _load_qags(QAGS_X_URL, "qags_x", "qags_x.jsonl")


def load_summeval() -> list[dict]:
    """Each row of mteb/summeval has 16 machine summaries + per-expert consistency lists.

    We flatten to one row per (doc, machine_summary) pair and binarize the mean
    expert consistency score at 4.0.
    """
    ds = load_dataset("mteb/summeval", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        article = ex["text"]
        machine_summaries = ex["machine_summaries"]
        consistency_scores = ex["consistency"]
        for j, (summary, score) in enumerate(zip(machine_summaries, consistency_scores)):
            if score is None:
                continue
            mean_score = float(score)
            rows.append(
                {
                    "id": f"summeval-{i}-{j}",
                    "benchmark": "summeval",
                    "source_text": article,
                    "response_text": summary,
                    "label": 0 if mean_score >= SUMMEVAL_CONSISTENT_THRESHOLD else 1,
                    "raw_score": mean_score,
                }
            )
    return rows


_LOADERS = {
    "summeval": load_summeval,
    "qags_c": load_qags_c,
    "qags_x": load_qags_x,
}


def load_benchmark(name: str, limit: Optional[int] = None) -> list[dict]:
    if name not in _LOADERS:
        raise ValueError(f"Unknown benchmark {name!r}; expected one of {sorted(_LOADERS)}")
    rows = _LOADERS[name]()
    if limit is not None:
        rows = rows[:limit]
    return rows


def available_benchmarks() -> Iterable[str]:
    return _LOADERS.keys()
