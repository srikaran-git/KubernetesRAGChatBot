import logfire
from typing import List


def _chunk_end(text: str, start: int, chunk_size: int) -> int:
    """Choose the last whitespace boundary in a chunk window when possible."""
    window_end = min(start + chunk_size, len(text))
    if window_end == len(text):
        return window_end

    # Prefer a natural boundary, but do not make a chunk much shorter merely
    # because an early space/newline happens to exist.
    search_start = start + (chunk_size // 2)
    boundary = max(
        text.rfind("\n", search_start, window_end),
        text.rfind(" ", search_start, window_end),
    )
    return boundary if boundary > start else window_end


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """Split text into character windows with a configurable sliding overlap."""
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = _chunk_end(text, start, chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            # Advance less than one full window so the final 150 characters of
            # the previous window become the beginning of the next one.
            start = end - chunk_overlap
            while start < text_length and text[start].isspace():
                start += 1

        logfire.info(
            f"✅ Generated {len(chunks)} chunks with {chunk_overlap}-character overlap"
        )

        return chunks
