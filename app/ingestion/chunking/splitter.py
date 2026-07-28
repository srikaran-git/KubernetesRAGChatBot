from typing import List
import logfire

from typing import List
import textwrap
import logfire

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    with logfire.span("✂️ Text Chunking", text_length=len(text)):

        if not text.strip():
            return []

        chunks = textwrap.wrap(
            text,
            width=chunk_size,
            break_long_words=False,
            replace_whitespace=False,
        )

        logfire.info(f"✅ Generated {len(chunks)} chunks")

        return chunks