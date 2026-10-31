"""Qdrant collection setup and hybrid document indexing."""
from __future__ import annotations

import uuid
from typing import Sequence

from qdrant_client import QdrantClient, models

from app.services.retrieval.embeddings import embed_texts
from app.services.retrieval.sparse_embeddings import embed_sparse_documents


def ensure_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
) -> None:
    """Create a named-vector collection if it does not already exist."""
    if dense_dim <= 0:
        raise ValueError("dense_dim must be positive")

    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        try:
            existing_dim = vectors["dense"].size
        except (KeyError, TypeError, AttributeError) as error:
            raise RuntimeError(
                f"Collection '{collection_name}' does not have a named dense vector. "
                "Recreate it with --wipe."
            ) from error

        if existing_dim != dense_dim:
            raise RuntimeError(
                f"Collection '{collection_name}' uses {existing_dim}-dimension dense vectors, "
                f"but the configured embedding model produces {dense_dim}. "
                "Recreate it with --wipe before ingesting."
            )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    # Qdrant Cloud requires payload indexes for the metadata filters used by
    # checkpoint verification and stale-chunk/source cleanup.
    for field_name, field_schema in (
        ("source", models.PayloadSchemaType.KEYWORD),
        ("source_type", models.PayloadSchemaType.KEYWORD),
        ("source_scope", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def point_id(source: str, chunk_index: int) -> str:
    """Stable IDs make repeated ingestion idempotent for the same source/chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hybrid-rag:{source}:{chunk_index}"))


def index_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: Sequence[str],
    source: str,
    source_type: str = "general",
    source_scope: str | None = None,
    batch_size: int = 32,
) -> int:
    """Embed and upsert chunks into both dense and sparse named vectors."""
    clean_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not clean_chunks:
        return 0
    if not source or not source.strip():
        raise ValueError("source is required")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    total = 0
    for start in range(0, len(clean_chunks), batch_size):
        batch = clean_chunks[start : start + batch_size]
        dense_vectors = embed_texts(batch)
        sparse_vectors = embed_sparse_documents(batch)

        if len(dense_vectors) != len(batch) or len(sparse_vectors) != len(batch):
            raise RuntimeError("Embedding count does not match chunk count")

        points = []
        for offset, (text, dense, sparse) in enumerate(
            zip(batch, dense_vectors, sparse_vectors)
        ):
            chunk_index = start + offset
            points.append(
                models.PointStruct(
                    id=point_id(source, chunk_index),
                    vector={
                        "dense": dense,
                        "sparse": models.SparseVector(
                            indices=sparse["indices"],
                            values=sparse["values"],
                        ),
                    },
                    payload={
                        "text": text,
                        "source": source,
                        "source_type": source_type,
                        "source_scope": source_scope or source_type,
                        "chunk_index": chunk_index,
                        "embedding_version": "gemini-embedding-001-1024d",
                        "embedding_dimension": len(dense),
                        "sparse_model": "Qdrant/bm25",
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total += len(points)

    # Remove stale tail chunks if a source was re-ingested with fewer chunks.
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        range=models.Range(gte=len(clean_chunks)),
                    ),
                ]
            )
        ),
        wait=True,
    )
    return total
