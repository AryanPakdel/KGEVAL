from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def triple_to_sentence(triple: dict) -> str:
    return f"{triple['entity1']} {triple['relation']} {triple['entity2']}"


def embed_sentences(sentences: list[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, _get_model().get_sentence_embedding_dimension()))
    return _get_model().encode(sentences, convert_to_numpy=True, show_progress_bar=False)


def embed_triples(triples: list[dict]) -> np.ndarray:
    return embed_sentences([triple_to_sentence(t) for t in triples])
