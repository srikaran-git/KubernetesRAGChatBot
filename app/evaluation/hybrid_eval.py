"""Small, dependency-free metrics for hybrid retrieval regression tests."""

from __future__ import annotations

from collections.abc import Iterable

def search_enterprise_knowledge(query: str, limit: int):
    """Import retrieval lazily so offline metric tests need no model runtime."""
    from app.services.retrieval.qdrant_service import search_enterprise_knowledge as search

    return search(query, limit=limit)


def evaluate(examples: Iterable[dict], k: int = 5) -> dict[str, float | int]:
    """Calculate source-level recall@k and reciprocal rank for query examples.

    Every example must contain ``query`` and ``relevant_sources``. Matching is
    performed against the retrieval result's ``source`` field, not chunk text.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    examples = list(examples)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for example in examples:
        query = str(example.get("query", "")).strip()
        relevant_sources = {
            str(source) for source in example.get("relevant_sources", []) if source
        }
        if not query:
            raise ValueError("Each evaluation example needs a non-empty query")
        if not relevant_sources:
            raise ValueError("Each evaluation example needs relevant_sources")

        results = search_enterprise_knowledge(query, limit=k)
        ranked_sources = [str(result.get("source", "")) for result in results[:k]]
        retrieved_relevant = relevant_sources.intersection(ranked_sources)
        recalls.append(len(retrieved_relevant) / len(relevant_sources))

        rank = next(
            (
                index
                for index, source in enumerate(ranked_sources, start=1)
                if source in relevant_sources
            ),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    count = len(examples)
    return {
        "examples": count,
        f"recall@{k}": sum(recalls) / count if count else 0.0,
        f"mrr@{k}": sum(reciprocal_ranks) / count if count else 0.0,
    }
