import pytest

from app.ingestion.chunking.splitter import chunk_text


def test_chunk_text_uses_requested_overlap():
    text = "a" * 1_000 + " " + "b" * 1_000

    chunks = chunk_text(text, chunk_size=1_000, chunk_overlap=150)

    assert len(chunks) >= 2
    assert chunks[0][-150:] == chunks[1][:150]


@pytest.mark.parametrize("overlap", [-1, 1_000])
def test_chunk_text_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("example text", chunk_size=1_000, chunk_overlap=overlap)
