from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.retrieval.hybrid_index import ensure_hybrid_collection, index_chunks


def test_ensure_collection_creates_named_dense_and_sparse_vectors():
    client = MagicMock()
    client.collection_exists.return_value = False

    ensure_hybrid_collection(client, "test_hybrid", dense_dim=3)

    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert set(kwargs["vectors_config"]) == {"dense"}
    assert set(kwargs["sparse_vectors_config"]) == {"sparse"}


def test_existing_collection_is_not_recreated():
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.config.params.vectors = {
        "dense": SimpleNamespace(size=3)
    }

    ensure_hybrid_collection(client, "test_hybrid", dense_dim=3)

    client.create_collection.assert_not_called()


@patch("app.services.retrieval.hybrid_index.embed_texts")
@patch("app.services.retrieval.hybrid_index.embed_sparse_documents")
def test_index_chunks_upserts_both_vectors(mock_sparse, mock_dense):
    mock_dense.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_sparse.return_value = [
        {"indices": [1], "values": [0.5]},
        {"indices": [2], "values": [0.8]},
    ]
    client = MagicMock()

    count = index_chunks(
        client=client,
        collection_name="test_hybrid",
        chunks=["first chunk", "second chunk"],
        source="manual.txt",
        source_type="docs",
    )

    assert count == 2
    client.upsert.assert_called_once()
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    assert set(points[0].vector.keys()) == {"dense", "sparse"}
    assert points[0].payload["source"] == "manual.txt"
    client.delete.assert_called_once()
