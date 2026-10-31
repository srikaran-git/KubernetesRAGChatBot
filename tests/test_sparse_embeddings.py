from types import SimpleNamespace

import pytest

from app.services.retrieval.sparse_embeddings import _as_sparse_payload, embed_sparse_query


def test_sparse_payload_normalizes_types():
    result = _as_sparse_payload(SimpleNamespace(indices=[1, 3], values=[0.25, 0.75]))
    assert result == {"indices": [1, 3], "values": [0.25, 0.75]}


def test_sparse_payload_rejects_wrong_shape():
    with pytest.raises(TypeError):
        _as_sparse_payload(object())


def test_empty_sparse_query_rejected():
    with pytest.raises(ValueError):
        embed_sparse_query("  ")
