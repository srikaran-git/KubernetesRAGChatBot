"""Sparse query/document vectors used by Qdrant hybrid retrieval."""
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding


SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    """Load the local sparse model once per process."""
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def as_sparse_payload(embedding: Any) -> dict[str, list]:
    """Convert FastEmbed SparseEmbedding into JSON/Qdrant-compatible fields."""
    indices = getattr(embedding, "indices", None)
    values = getattr(embedding, "values", None)
    if indices is None or values is None:
        raise TypeError("Sparse embedding must expose indices and values")
    return {
        "indices": [int(i) for i in indices],
        "values": [float(v) for v in values],
    }


def embed_sparse_query(text: str) -> dict[str, list]:
    if not text or not text.strip():
        raise ValueError("Query cannot be empty")
    embedding = next(get_sparse_model().query_embed(text))
    return as_sparse_payload(embedding)


def embed_sparse_documents(texts: list[str]) -> list[dict[str, list]]:
    if not texts:
        return []
    model = get_sparse_model()
    return [as_sparse_payload(item) for item in model.passage_embed(texts)]
