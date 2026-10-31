"""Hybrid Qdrant retrieval using dense + sparse prefetch and RRF fusion."""
import logfire
from qdrant_client import QdrantClient, models

from app.config import settings
from app.services.retrieval.embeddings import embed_query
from app.services.retrieval.sparse_embeddings import embed_sparse_query


client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
HYBRID_COLLECTION = getattr(
    settings, "QDRANT_HYBRID_COLLECTION", "enterprise_rag_hybrid"
)


def format_points(points) -> list[dict]:
    results = []
    for point in points:
        payload = point.payload or {}
        results.append(
            {
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "source_type": payload.get("source_type", "general"),
                "chunk_index": payload.get("chunk_index"),
                "score": float(point.score) if point.score is not None else 0.0,
            }
        )
    return results


def search_enterprise_knowledge(query: str, limit: int = 8) -> list[dict]:
    """Run dense and sparse retrieval, then fuse candidate ranks with Qdrant RRF."""
    if not query or not query.strip():
        return []
    if limit < 1:
        raise ValueError("limit must be at least 1")

    try:
        dense_query = embed_query(query)
        sparse_query = embed_sparse_query(query)

        response = client.query_points(
            collection_name=HYBRID_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=max(20, limit * 4),
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using="sparse",
                    limit=max(20, limit * 4),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return format_points(response.points)

    except Exception as hybrid_error:
        # Graceful degradation: if sparse inference/fusion fails, still try dense.
        logfire.error(f"Hybrid Qdrant search failed: {hybrid_error}")
        try:
            dense_query = embed_query(query)
            response = client.query_points(
                collection_name=HYBRID_COLLECTION,
                query=dense_query,
                using="dense",
                limit=limit,
                with_payload=True,
            )
            logfire.warning("Returned dense-only fallback results.")
            return format_points(response.points)
        except Exception as dense_error:
            logfire.error(f"Dense fallback search failed: {dense_error}")
            return []
