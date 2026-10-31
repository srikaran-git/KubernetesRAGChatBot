# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Live Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Calls the running FastAPI /query endpoint for each golden sample.
# Captures actual_response, actual_contexts and actual_tools_called.
#
# Checkpointing: every successful response is saved to disk immediately
# (evals/results/pipeline_results.json). Re-running resumes: samples that already
# have a response are skipped, only missing / failed ones are called again.
#
# Guardrails evaluation has intentionally been removed from this pipeline.
# The request asks the backend to cap LLM output at 2048 tokens.
# ─────────────────────────────────────────────────────────────────────────────
 
import time
import copy
import json
import os
import requests
import logfire
 
from evals.checkpoint import (
    load_pipeline_samples,
    save_pipeline_sample,
    clear_pipeline_samples,
)
 
 
API_URL = "http://localhost:8000/query"
 
# Maximum number of tokens the backend LLM should generate.
MAX_OUTPUT_TOKENS = 2048
 
# Keep the stored answer short for the later RAGAS evaluation.
RESPONSE_TRUNCATE = 300
 
# Delay between live requests to reduce pressure on the production API key.
DELAY_BETWEEN_CALLS = 10
 
REQUEST_TIMEOUT = 120
 
_SAVED_FIELDS = ("actual_response", "actual_contexts", "actual_tools_called")
 
 
def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from the /query response to a tool name.
 
    Guardrails are intentionally NOT handled here.
    """
    joined = " ".join(thought_process).lower()
 
    if (
        "intent: technical" in joined
        or "search term:" in joined
        or "context retrieved" in joined
    ):
        return "retrieve_documents"
 
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
 
    return "unknown"
 
 
def _sid(sample: dict, i: int) -> str:
    return str(sample.get("id", i))
 
 
def _query_one(sample: dict, i: int, n: int) -> bool:
    """Calls /query for one sample. Fills the sample in place. True on success."""
    question = sample["question"]
 
    with logfire.span(
        f"📤 Live Query {i + 1}/{n}",
        question=question[:80],
        domain=sample.get("domain", ""),
    ):
        try:
            payload = {
                "q": question,
                "thread_id": f"eval_run_{i}",
                "max_tokens": MAX_OUTPUT_TOKENS,
            }
 
            resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
 
            # A 429 from the backend is a real rate-limit failure.
            # Do not silently convert it into a successful evaluation.
            if resp.status_code == 429:
                raise RuntimeError(
                    "FastAPI/LLM rate limit (HTTP 429). "
                    "Configure fallback keys in the FastAPI LLM client."
                )
 
            resp.raise_for_status()
            data = resp.json()
 
            raw_answer = data.get("answer") or ""
            thought_process = data.get("thought_process") or []
            sources = data.get("sources") or []
 
            sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
            sample["actual_contexts"] = sources[:5]
            sample["actual_tools_called"] = [detect_tool(thought_process)]
 
            logfire.info(
                "✅ Response captured",
                tool=sample["actual_tools_called"][0],
                response_chars=len(raw_answer),
                context_chunks=len(sources),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            return bool(sample["actual_response"])
 
        except requests.exceptions.ConnectionError:
            logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
        except Exception as e:
            logfire.error(f"❌ Query failed: {e}")
 
        sample["actual_response"] = ""
        sample["actual_contexts"] = sample.get("relevant_contexts", [])
        sample["actual_tools_called"] = ["unknown"]
        return False
 
 
def run_pipeline(golden_dataset: dict, progress_callback=None, resume: bool = True) -> dict:
    """
    Enrich every RAG sample with a live API response.
 
    resume=True  → reuse responses saved by an earlier (possibly interrupted)
                   run; only missing / failed samples hit the API.
    resume=False → discard saved responses and start fresh.
 
    Returns a deep copy containing:
      - actual_response
      - actual_contexts
      - actual_tools_called
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)
 
    if not resume:
        clear_pipeline_samples()
 
    saved = load_pipeline_samples() if resume else {}
 
    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]
 
            if progress_callback:
                progress_callback(i, n, question, "calling")
 
            cached = saved.get(_sid(sample, i))
            from_cache = bool(
                cached
                and cached.get("question") == question
                and cached.get("actual_response")
            )
 
            if from_cache:
                for field in _SAVED_FIELDS:
                    sample[field] = cached[field]
                logfire.info("⏭️ Reused saved response", question=question[:80])
            else:
                ok = _query_one(sample, i, n)
                if ok:
                    save_pipeline_sample(
                        _sid(sample, i),
                        {"question": question, **{f: sample[f] for f in _SAVED_FIELDS}},
                    )   # ← checkpoint
 
            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])
 
            # Only throttle after real API calls.
            if not from_cache and i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)
 
    return dataset
 
 
def load_saved_dataset(golden_dataset: dict) -> tuple[dict, int]:
    """
    Rebuilds an enriched dataset from saved responses without calling the API.
    Returns (dataset, number_of_samples_that_have_a_saved_response).
    """
    dataset = copy.deepcopy(golden_dataset)
    saved = load_pipeline_samples()
    ok = 0
 
    for i, sample in enumerate(dataset["rag_samples"]):
        cached = saved.get(_sid(sample, i))
        if cached and cached.get("question") == sample["question"] and cached.get("actual_response"):
            for field in _SAVED_FIELDS:
                sample[field] = cached[field]
            ok += 1
 
    return dataset, ok
 
 
def count_saved(golden_dataset: dict) -> int:
    return load_saved_dataset(golden_dataset)[1]
 
 
def save_results(dataset: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
 
 
def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
 
    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)
 