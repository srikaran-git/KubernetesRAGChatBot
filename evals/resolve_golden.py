"""
Maps each gold context in golden_dataset.json to the real chunks in Qdrant.

Run from the project root:   python evals/resolve_golden.py
Output:     evals/golden_dataset.resolved.json  (adds gold_chunks per context)
Also prints contexts that match NO chunk -> these are either not ingested
or paraphrased, and will always score 0 until fixed.
"""
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()  # finds .env in this folder or any parent folder

# ---- CONFIG ----
HERE = Path(__file__).resolve().parent          # the evals/ folder
IN_FILE = HERE / "golden_dataset.json"
OUT_FILE = HERE / "golden_dataset.resolved.json"
TEXT_KEY = "text"                               # payload field holding the chunk text
DEFAULT_COLLECTION = os.getenv("QDRANT_HYBRID_COLLECTION", "ENTERPRISE_RAG_CHATBOT")
# ----------------


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        sys.exit(
            "QDRANT_CLUSTER_ENDPOINT and/or QDRANT_API_KEY missing. "
            "Add them to your .env file."
        )
    return QdrantClient(url=url.rstrip("/"), api_key=api_key)


def load_chunks(client: QdrantClient, collection: str):
    chunks, offset = [], None
    while True:
        pts, offset = client.scroll(
            collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in pts:
            pl = p.payload or {}
            chunks.append({
                "point_id": str(p.id),
                "source": pl.get("source"),
                "chunk_index": pl.get("chunk_index"),
                "text": norm(pl.get(TEXT_KEY, "")),   # normalized once here
            })
        if offset is None:
            break
    return chunks


def is_relevant(chunk_text: str, ctx: dict) -> bool:
    """Use this same function in your eval metrics (pass raw chunk text)."""
    t = norm(chunk_text)
    return any(norm(a) in t for a in ctx["must_contain"])


def main():
    collection = os.getenv(
        "QDRANT_HYBRID_COLLECTION",
        os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )
    client = get_client()

    # Fail early with a clear message if auth or collection access is wrong
    print("Collections:", [c.name for c in client.get_collections().collections])
    print("Points in collection:", client.count(collection, exact=True).count)

    chunks = load_chunks(client, collection)
    print(f"Loaded {len(chunks)} chunks from '{collection}'")

    data = json.loads(IN_FILE.read_text(encoding="utf-8"))
    missing = []

    for sample in data["rag_samples"]:
        for ci, ctx in enumerate(sample["relevant_contexts"]):
            anchors = [norm(a) for a in ctx["must_contain"]]
            hits = [c for c in chunks if any(a in c["text"] for a in anchors)]
            ctx["gold_chunks"] = [
                {"point_id": h["point_id"], "source": h["source"],
                 "chunk_index": h["chunk_index"]} for h in hits
            ]
            if hits:
                ctx["source"] = hits[0]["source"]
                ctx["chunk_index"] = hits[0]["chunk_index"]
            else:
                missing.append((sample["id"], ci, sample["question"]))

    OUT_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(s["relevant_contexts"]) for s in data["rag_samples"])
    print(f"Resolved {total - len(missing)}/{total} gold contexts -> {OUT_FILE}")
    if missing:
        print("\nNO matching chunk found (not ingested, or text differs from corpus):")
        for sid, ci, q in missing:
            print(f"  sample {sid} context #{ci}: {q}")


if __name__ == "__main__":
    main()
