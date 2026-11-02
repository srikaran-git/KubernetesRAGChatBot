"""
Adapters that let eval_retrieval.py score your REAL pipeline.

Save as evals/app_adapter.py, then run from the project root:

    # hybrid retrieval only (compares your app's embeddings/search to the rebuilt ones)
    python evals/eval_retrieval.py --verbose --app-func evals.app_adapter:retrieve_no_rerank

    # full pipeline: hybrid retrieval + FlashRank (what retrieve_node actually does)
    python evals/eval_retrieval.py --verbose --app-func evals.app_adapter:retrieve

Both return Qdrant point IDs in final ranked order.
"""
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def _candidates(question: str, pool: int):
    """Same candidate building as app/agent/nodes/retriever.py."""
    raw = search_enterprise_knowledge(question, limit=pool)
    if not raw:
        print(f"[adapter] EMPTY retrieval (search swallowed an error?): {question!r}")
    content_to_id, candidates = {}, []
    for r in raw:
        content = r.get("content", "").strip()
        if content and content not in content_to_id:
            content_to_id[content] = r["point_id"]
            candidates.append(content)
    return raw, content_to_id, candidates


def retrieve_no_rerank(question: str, k: int):
    raw, _, _ = _candidates(question, k)
    return [r["point_id"] for r in raw]


def retrieve(question: str, k: int):
    raw, content_to_id, candidates = _candidates(question, k)
    # top_n = everything, so the eval can score the first 5 AND the full pool.
    # The first 5 are identical to what the app returns with top_n=5.
    reranked = rerank_documents(question, candidates, top_n=len(candidates))
    ids = [content_to_id[c] for c in reranked if c in content_to_id]
    if len(ids) != len(reranked):
        print(f"[adapter] {len(reranked) - len(ids)} reranked passages lost their ID mapping: {question!r}")
    if reranked == candidates[: len(reranked)] and len(candidates) > 1:
        print(f"[adapter] reranker returned the input order unchanged (fallback or no reordering): {question!r}")
    return ids