import logfire

from app.agent.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """Retrieve with hybrid Qdrant RRF, then rerank with existing FlashRank."""
    query = state["current_query"]

    with logfire.span("Hybrid Knowledge Retrieval"):
        logfire.info(f"Hybrid search query: {query}")
        raw_results = search_enterprise_knowledge(query, limit=20)
        logfire.info(f"Hybrid retrieval returned {len(raw_results)} candidates")

        # Keep a mapping so the reranker can return text while source metadata survives.
        content_to_result = {}
        candidates = []
        for result in raw_results:
            content = result.get("content", "").strip()
            if content and content not in content_to_result:
                content_to_result[content] = result
                candidates.append(content)

        with logfire.span("FlashRank Reranking"):
            reranked_contents = rerank_documents(query, candidates, top_n=5)

        formatted_docs = []
        for content in reranked_contents:
            metadata = content_to_result.get(content, {})
            source = metadata.get("source", "Unknown")
            chunk_index = metadata.get("chunk_index")
            source_label = f"{source}" + (
                f" (chunk {chunk_index})" if chunk_index is not None else ""
            )
            formatted_docs.append(
                f"SOURCE: {source_label}\nCONTENT:\n{content}"
            )

    return {
        "documents": formatted_docs,
        "status": f"Hybrid retrieval found {len(raw_results)} candidates.",
        "plan": state["plan"] + ["Hybrid Retrieval + RRF", "FlashRank Reranking"],
    }
