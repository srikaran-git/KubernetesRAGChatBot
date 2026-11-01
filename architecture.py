"""Generate a Mermaid diagram of the Enterprise Agentic RAG architecture.

Usage:
    python architecture.py
    python architecture.py --output architecture.mmd

Paste the emitted Mermaid text into GitHub, Mermaid Live Editor, or a Markdown
document that supports Mermaid to render the diagram.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def get_mermaid_diagram() -> str:
    """Return the current system architecture as a Mermaid flowchart."""
    return """flowchart LR
    user([User]) --> ui[Streamlit UI]
    ui -->|POST /query| api[FastAPI API]

    api --> rails{NeMo Guardrails}
    rails -->|Blocked| blocked[Safe response]
    blocked --> ui

    rails -->|Allowed| workflow_node[LangGraph RAG workflow]
    workflow_node --> planner[Planner]
    planner -->|Conversational| responder[Responder]
    planner -->|Technical| retriever[Hybrid retriever]

    retriever --> dense[Gemini dense embeddings]
    retriever --> sparse[FastEmbed BM25 sparse embeddings]
    dense --> qdrant[(Qdrant hybrid collection)]
    sparse --> qdrant
    qdrant --> rrf[RRF fusion]
    rrf --> reranker[FlashRank reranker]
    reranker --> responder

    responder --> gateway[Portkey gateway]
    gateway --> llm[Configured LLM provider]
    llm --> responder
    responder --> api

    api --> memory[(LangGraph MemorySaver)]
    api --> ui

    ingest[Ingestion CLI] --> parsers[PDF / HTML / text / Office parsers]
    parsers --> chunking[Text chunking]
    chunking --> dense
    chunking --> sparse

    api -. traces .-> observability[Logfire / LangSmith]
    ingest -. traces .-> observability
    evals[Evaluation app] -->|calls /query| api
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print or save the Enterprise Agentic RAG Mermaid architecture diagram."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the generated .mmd Mermaid file.",
    )
    args = parser.parse_args()

    diagram = get_mermaid_diagram()
    if args.output:
        args.output.write_text(diagram, encoding="utf-8")
        print(f"Architecture diagram written to {args.output}")
    else:
        print(diagram)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
