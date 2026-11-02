"""
Stage-by-stage retrieval eval for the hybrid Qdrant RAG.

Scores dense, sparse, hybrid (RRF) and optionally reranked results against the
gold chunk IDs in golden_dataset.resolved.json, so you can see exactly which
stage loses the gold chunk.

Usage (from project root):
    python evals/eval_retrieval.py
    python evals/eval_retrieval.py --rerank BAAI/bge-reranker-base
    python evals/eval_retrieval.py --app-func app.services.retrieval.qdrant_service:search_enterprise_knowledge
    python evals/eval_retrieval.py --skip-self-test --k 5 --pool 15

Install:
    pip install -r requirements.txt
    pip install sentence-transformers      # only if you use --rerank

.env needs: QDRANT_CLUSTER_ENDPOINT, QDRANT_API_KEY, and GEMINI_API_KEY.
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so --app-func can import your project code

DEFAULT_FILE = HERE / "golden_dataset.resolved.json"
DEFAULT_COLLECTION = os.getenv("QDRANT_HYBRID_COLLECTION", "ENTERPRISE_RAG_CHATBOT")
DENSE_NAME, SPARSE_NAME = "dense", "sparse"   # named vectors seen in your payload
EMBED_MODEL, EMBED_DIM = "gemini-embedding-001", 1024
PREFETCH_LIMIT = 50


# ---------- embedders ----------
def make_dense_embedder():
    """Use the application embedder so eval vectors exactly match production."""
    from app.services.retrieval.embeddings import embed_query

    return embed_query


def make_sparse_embedder():
    """Use the application sparse embedder so model configuration cannot drift."""
    from app.services.retrieval.sparse_embeddings import embed_sparse_query

    def embed(q: str):
        payload = embed_sparse_query(q)
        return models.SparseVector(indices=payload["indices"], values=payload["values"])

    return embed


def make_reranker(name: str):
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(name)

    def rerank(q, points):
        scores = m.predict([(q, (p.payload or {}).get("text", "")) for p in points])
        order = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)
        return [points[i] for i in order]

    return rerank


# ---------- searches ----------
def dense_search(client, coll, vec, n):
    return client.query_points(coll, query=vec, using=DENSE_NAME, limit=n,
                               with_payload=True).points


def sparse_search(client, coll, svec, n):
    return client.query_points(coll, query=svec, using=SPARSE_NAME, limit=n,
                               with_payload=True).points


def hybrid_search(client, coll, vec, svec, n):
    return client.query_points(
        coll,
        prefetch=[
            models.Prefetch(query=vec, using=DENSE_NAME, limit=PREFETCH_LIMIT),
            models.Prefetch(query=svec, using=SPARSE_NAME, limit=PREFETCH_LIMIT),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=n, with_payload=True,
    ).points


# ---------- metrics ----------
def gold_ids(sample):
    return {g["point_id"] for c in sample["relevant_contexts"] for g in c["gold_chunks"]}


def first_rank(ids, gold):
    for r, pid in enumerate(ids, 1):
        if pid in gold:
            return r
    return None


def metrics(ids, gold, k):
    top = ids[:k]
    hits = [1 if pid in gold else 0 for pid in top]
    recall = len(gold & set(top)) / len(gold) if gold else 0.0
    precision = sum(hits) / k
    rank = first_rank(top, gold)
    mrr = 1 / rank if rank else 0.0
    return recall, precision, mrr


# ---------- optional: your own retriever ----------
def load_app_func(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def to_id(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("point_id", "id"):
            if key in x:
                return str(x[key])
    if hasattr(x, "id"):
        return str(x.id)
    raise ValueError("Your retriever must return point ids (str), dicts with 'id'/'point_id', or objects with .id")


# ---------- self test ----------
def self_test(client, coll, dense_e, sparse_e, data):
    """Use a chunk's own text as the query: it should come back at rank 1."""
    print("\n== Self-retrieval test (a chunk's own text should rank #1) ==")
    ctx = data["rag_samples"][0]["relevant_contexts"][0]
    pid = ctx["gold_chunks"][0]["point_id"]
    pt = client.retrieve(coll, ids=[pid], with_payload=True)[0]
    q = pt.payload["text"][:400]
    vec, svec = dense_e(q), sparse_e(q)
    for name, res in [("dense", dense_search(client, coll, vec, 5)),
                      ("sparse", sparse_search(client, coll, svec, 5)),
                      ("hybrid", hybrid_search(client, coll, vec, svec, 5))]:
        ids = [str(p.id) for p in res]
        r = first_rank(ids, {pid})
        print(f"  {name:<7} rank of own chunk: {r if r else 'NOT IN TOP 5  <-- problem'}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=15, help="candidates retrieved per stage")
    ap.add_argument("--rerank", default=None, help="cross-encoder model name")
    ap.add_argument("--app-func", default=None,
                    help="module:function. Called as f(question, k) -> list of point ids")
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print per-question ranks")
    a = ap.parse_args()

    url, key = os.getenv("QDRANT_CLUSTER_ENDPOINT"), os.getenv("QDRANT_API_KEY")
    if not url or not key:
        sys.exit("QDRANT_CLUSTER_ENDPOINT / QDRANT_API_KEY missing in .env")
    coll = os.getenv(
        "QDRANT_HYBRID_COLLECTION",
        os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )
    client = QdrantClient(url=url.rstrip("/"), api_key=key)

    data = json.loads(Path(a.file).read_text(encoding="utf-8"))
    samples = data["rag_samples"]
    for s in samples:
        for c in s["relevant_contexts"]:
            if "gold_chunks" not in c:
                sys.exit("Run resolve_golden.py first (gold_chunks missing).")

    dense_e, sparse_e = make_dense_embedder(), make_sparse_embedder()
    reranker = make_reranker(a.rerank) if a.rerank else None
    app_fn = load_app_func(a.app_func) if a.app_func else None

    if not a.skip_self_test:
        self_test(client, coll, dense_e, sparse_e, data)

    stages = ["dense", "sparse", "hybrid"]
    if reranker:
        stages.append("hybrid+rerank")
    if app_fn:
        stages.append("YOUR_APP")
    totals = {s: [0.0, 0.0, 0.0, 0.0] for s in stages}  # recall@k, prec@k, mrr@k, recall@pool
    n = 0

    print(f"\n== Per-question rank of first gold chunk (pool={a.pool}) ==" if a.verbose else "")
    for s in samples:
        gold = gold_ids(s)
        if not gold:
            continue
        n += 1
        q = s["question"]
        vec, svec = dense_e(q), sparse_e(q)

        results = {
            "dense": dense_search(client, coll, vec, a.pool),
            "sparse": sparse_search(client, coll, svec, a.pool),
            "hybrid": hybrid_search(client, coll, vec, svec, a.pool),
        }
        if reranker:
            results["hybrid+rerank"] = reranker(q, results["hybrid"])
        ids_by_stage = {k: [str(p.id) for p in v] for k, v in results.items()}
        if app_fn:
            ids_by_stage["YOUR_APP"] = [to_id(x) for x in app_fn(q, a.pool)]

        line = f"  Q{s['id']:<3}"
        for st in stages:
            ids = ids_by_stage[st]
            r, p, m = metrics(ids, gold, a.k)
            pool_recall = len(gold & set(ids[:a.pool])) / len(gold)
            for i, v in enumerate((r, p, m, pool_recall)):
                totals[st][i] += v
            rk = first_rank(ids, gold)
            line += f" | {st}: {rk if rk else '-':>2}"
        if a.verbose:
            print(line)

    print(f"\n== Averages over {n} questions (k={a.k}, pool={a.pool}) ==")
    print(f"{'stage':<15}{'Recall@'+str(a.k):>10}{'Prec@'+str(a.k):>10}{'MRR@'+str(a.k):>10}{'Recall@'+str(a.pool):>11}")
    for st in stages:
        r, p, m, pr = (v / n for v in totals[st])
        print(f"{st:<15}{r:>10.3f}{p:>10.3f}{m:>10.3f}{pr:>11.3f}")

    print("\nHow to read it:")
    print("  all stages ~0          -> query embedding / vector-name / collection mismatch")
    print("  dense ok, sparse ~0    -> sparse vector name or BM25 query encoding")
    print("  hybrid ok, rerank low  -> reranker bug (order, text field, slice)")
    print("  all stages fine, YOUR_APP ~0 -> bug is in your app's retrieval or eval wiring")


if __name__ == "__main__":
    main()
