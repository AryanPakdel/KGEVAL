from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity

from matching.embedder import embed_triples


def match_triples(
    response_triples: list[dict], source_triples: list[dict]
) -> list[dict]:
    """For each response triple, find the best-matching source triple.

    Returns a list with one entry per response triple:
        {
          "response_triple": {...},
          "best_source_triple": {...} | None,
          "similarity": float,         # 0.0 when no source triples exist
        }
    """
    results: list[dict] = []

    if not source_triples:
        for r in response_triples:
            results.append(
                {"response_triple": r, "best_source_triple": None, "similarity": 0.0}
            )
        return results

    source_embeddings = embed_triples(source_triples)
    response_embeddings = embed_triples(response_triples)

    if len(response_triples) == 0:
        return results

    sim_matrix = cosine_similarity(response_embeddings, source_embeddings)

    for i, r in enumerate(response_triples):
        row = sim_matrix[i]
        best_idx = int(row.argmax())
        best_score = float(row[best_idx])
        results.append(
            {
                "response_triple": r,
                "best_source_triple": source_triples[best_idx],
                "similarity": best_score,
            }
        )

    return results
