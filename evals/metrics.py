# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — RAGAS + Tool Correctness metrics
# ─────────────────────────────────────────────────────────────────────────────
#
# • One public function per metric:  run_faithfulness(), run_answer_relevancy(), ...
# • Every scored sample is saved to disk immediately (evals/results/metrics/*.json)
#   so a crash / rate-limit / closed browser never loses finished work.
# • A metric that is already complete is SKIPPED and the remaining metrics run.
#   A half-finished metric RESUMES from the first unscored sample.
# • One metric failing (e.g. every Groq key rate-limited) does not stop the others.
# ─────────────────────────────────────────────────────────────────────────────
 
import os
import asyncio
import logfire
import pandas as pd
 
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
 
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)
 
from evals.checkpoint import (
    fingerprint,
    now_iso,
    load_metric_checkpoint,
    save_metric_checkpoint,
)
 
 
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)
 
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 
# Hard cap for judge-model output.
MAX_OUTPUT_TOKENS = 4096
 
COOLDOWN_STANDARD = 62   # between two LLM metrics
COOLDOWN_MINI = 40       # between two samples of the same metric
CONTEXT_TRUNCATE = 300
CONTEXT_LIMIT = 2
 
# Evaluation-only Groq rate-limit handling.
RATE_LIMIT_BACKOFF = (5, 10)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Groq key handling
# ─────────────────────────────────────────────────────────────────────────────
 
def get_groq_keys() -> list[str]:
    """
    Reads the judge key plus numbered fallback keys.
 
        JUDGE_GROQ_API_KEY, JUDGE_GROQ_API_KEY_2, JUDGE_GROQ_API_KEY_3, ...
    and, as a fallback:
        GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
    """
    keys = []
 
    for prefix in ("JUDGE_GROQ_API_KEY", "GROQ_FALLBACK_API_KEY"):
        first = os.getenv(prefix)
        if first and first.startswith("gsk_") and first not in keys:
            keys.append(first)
 
        for i in range(2, 20):
            key = os.getenv(f"{prefix}_{i}")
            if key and key.startswith("gsk_") and key not in keys:
                keys.append(key)
 
    return keys
 
 
class MetricRateLimited(RuntimeError):
    """Raised when every judge key is rate-limited for the current sample."""
 
 
class JudgeKeyManager:
    """Manages Groq judge keys and rotates on rate-limit/quota errors."""
 
    def __init__(self, keys: list[str], status_cb=None):
        if not keys:
            raise RuntimeError(
                "No Groq judge API keys found. Add JUDGE_GROQ_API_KEY / "
                "JUDGE_GROQ_API_KEY_2 / ... or GROQ_API_KEY / GROQ_API_KEY_2 / ... "
                "to your config/.env."
            )
        self.keys = keys
        self.index = 0
        self.status_cb = status_cb
 
    @property
    def current_key(self):
        return self.keys[self.index]
 
    @staticmethod
    def is_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(x in msg for x in (
            "429", "rate limit", "rate-limit", "ratelimit", "resource_exhausted",
            "resource exhausted", "quota exceeded", "quotaexceeded", "too many requests",
        ))
 
    def reset(self):
        """Start again from the first key (limits recover during cooldowns)."""
        self.index = 0
 
    def rotate(self) -> bool:
        if self.index >= len(self.keys) - 1:
            return False
        old = self.index + 1
        self.index += 1
        msg = f"⚠️ Groq judge rate limit hit. Switching evaluation key {old} → {self.index + 1}."
        logfire.warning(msg)
        if self.status_cb:
            self.status_cb(msg)
        return True
 
    def build_llm(self):
        client = AsyncOpenAI(api_key=self.current_key, base_url=GROQ_BASE_URL)
        return llm_factory(JUDGE_MODEL, provider="openai", client=client)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Run context — shared by all metric functions
# ─────────────────────────────────────────────────────────────────────────────
 
class RunContext:
    """
    Carries callbacks + lazily-built judge resources.
 
    The judge keys / embedding model are only created the first time a metric
    actually needs them, so re-opening a finished evaluation costs nothing.
    """
 
    def __init__(
        self,
        status_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        force: bool = False,
    ):
        self.status_cb = status_cb
        self.progress_cb = progress_cb      # (metric_key, done, total)
        self.force = force                  # ignore saved checkpoints
        self.made_llm_calls = False         # drives the cooldown between metrics
        self._manager = None
        self._embeddings = None
 
    @property
    def manager(self) -> JudgeKeyManager:
        if self._manager is None:
            self._manager = JudgeKeyManager(get_groq_keys(), self.status_cb)
        return self._manager
 
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL, use_api=False)
        return self._embeddings
 
    def say(self, msg: str):
        if self.status_cb:
            self.status_cb(msg)
 
 
async def cooldown(seconds: int, label: str, status_cb=None):
    if status_cb:
        status_cb(f"⏳ {seconds}s cooldown after {label}...")
 
    await asyncio.sleep(seconds)
 
    if status_cb:
        status_cb("✅ Ready — continuing evaluation.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Sample preparation
# ─────────────────────────────────────────────────────────────────────────────
 
def prep_samples(golden_dataset: dict) -> list:
    valid = []
 
    for s in golden_dataset["rag_samples"]:
        response = (s.get("actual_response") or "").strip()
 
        if not response:
            continue
 
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        # Golden contexts are now structured objects with text, anchors, and
        # resolved Qdrant IDs. Live API contexts remain strings. RAGAS accepts
        # text only, so normalize both representations at this boundary.
        contexts = [
            (str(context.get("text", "")) if isinstance(context, dict) else str(context))
            [:CONTEXT_TRUNCATE]
            for context in raw_contexts[:CONTEXT_LIMIT]
        ]
        contexts = [context for context in contexts if context]
 
        valid.append({**s, "actual_contexts": contexts})
 
    return valid
 
 
def _sample_id(s: dict, idx: int) -> str:
    return str(s.get("id", idx))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    build_inputs: Callable[[dict], dict]
    factory: Optional[Callable] = None     # (llm, embeddings) -> RAGAS metric; None = no LLM
    needs_embeddings: bool = False
    compute: Optional[Callable] = None     # inputs -> float, for non-LLM metrics
 
    @property
    def uses_llm(self) -> bool:
        return self.factory is not None
 
 
def _tool_score(inp: dict) -> float:
    called, expected = set(inp["called"]), set(inp["expected"])
    union = len(called | expected)
    return len(called & expected) / union if union else 0.0
 
 
SPECS: dict[str, MetricSpec] = {
    "faithfulness": MetricSpec(
        key="faithfulness",
        title="Faithfulness",
        build_inputs=lambda s: {
            "user_input": s["question"],
            "response": s["actual_response"],
            "retrieved_contexts": s["actual_contexts"],
        },
        factory=lambda llm, emb: Faithfulness(llm=llm),
    ),
    "answer_relevancy": MetricSpec(
        key="answer_relevancy",
        title="Answer Relevancy",
        build_inputs=lambda s: {
            "user_input": s["question"],
            "response": s["actual_response"],
        },
        factory=lambda llm, emb: AnswerRelevancy(llm=llm, embeddings=emb),
        needs_embeddings=True,
    ),
    "context_precision": MetricSpec(
        key="context_precision",
        title="Context Precision",
        build_inputs=lambda s: {
            "user_input": s["question"],
            "reference": s["reference"],
            "retrieved_contexts": s["actual_contexts"],
        },
        factory=lambda llm, emb: ContextPrecision(llm=llm),
    ),
    "context_recall": MetricSpec(
        key="context_recall",
        title="Context Recall",
        build_inputs=lambda s: {
            "user_input": s["question"],
            "reference": s["reference"],
            "retrieved_contexts": s["actual_contexts"],
        },
        factory=lambda llm, emb: ContextRecall(llm=llm),
    ),
    "answer_correctness": MetricSpec(
        key="answer_correctness",
        title="Answer Correctness",
        build_inputs=lambda s: {
            "user_input": s["question"],
            "response": s["actual_response"],
            "reference": s["reference"],
        },
        factory=lambda llm, emb: AnswerCorrectness(llm=llm, embeddings=emb),
        needs_embeddings=True,
    ),
    "tool_correctness": MetricSpec(
        key="tool_correctness",
        title="Tool Correctness",
        build_inputs=lambda s: {
            "called": sorted(set(s.get("actual_tools_called") or [])),
            "expected": sorted(set(s.get("expected_tools") or [])),
        },
        compute=_tool_score,   # zero LLM calls
    ),
}
 
# Tool correctness is free and instant, so it runs first and gives results at once.
RUN_ORDER = ["tool_correctness", "faithfulness", "answer_relevancy",
             "context_precision", "context_recall", "answer_correctness"]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint-aware core
# ─────────────────────────────────────────────────────────────────────────────
 
def _fp(spec: MetricSpec, inputs: dict) -> str:
    """Fingerprint of everything that influences a score."""
    return fingerprint({"inputs": inputs, "judge": JUDGE_MODEL if spec.uses_llm else None})
 
 
def _plan(spec: MetricSpec, samples: list, force: bool = False):
    """
    Splits samples into (already scored rows, work still to do) by comparing
    the saved fingerprints with the current inputs.
    """
    saved = {} if force else load_metric_checkpoint(spec.key).get("rows", {})
    valid, todo = {}, []
 
    for idx, s in enumerate(samples):
        sid = _sample_id(s, idx)
        inp = spec.build_inputs(s)
        fp = _fp(spec, inp)
        row = saved.get(sid)
 
        if row and row.get("fp") == fp and row.get("score") is not None:
            valid[sid] = row
        else:
            todo.append((sid, s, inp, fp))
 
    return valid, todo
 
 
def _rows_df(spec: MetricSpec, samples: list, rows: dict) -> pd.DataFrame:
    out = []
    for idx, s in enumerate(samples):
        row = rows.get(_sample_id(s, idx))
        if row:
            out.append({
                "id": s.get("id", idx),
                "question": s["question"][:65],
                spec.key: round(float(row["score"]), 3),
            })
    return pd.DataFrame(out, columns=["id", "question", spec.key])
 
 
async def _score_one(spec: MetricSpec, inputs: dict, ctx: RunContext) -> float:
    """Scores one sample with backoff + key rotation on rate limits."""
    manager = ctx.manager
    emb = ctx.embeddings if spec.needs_embeddings else None
    metric = spec.factory(manager.build_llm(), emb)
 
    while True:
        for retry in range(len(RATE_LIMIT_BACKOFF) + 1):
            ctx.made_llm_calls = True
            try:
                results = await metric.abatch_score([inputs])
                value = float(results[0].value)
                if value != value:  # NaN
                    raise ValueError("judge returned NaN")
                return value
            except Exception as exc:
                if not manager.is_rate_limit(exc):
                    raise
                if retry < len(RATE_LIMIT_BACKOFF):
                    wait = RATE_LIMIT_BACKOFF[retry]
                    ctx.say(f"⏳ {spec.title}: rate limit — retry {retry + 1}/"
                            f"{len(RATE_LIMIT_BACKOFF)} in {wait}s...")
                    await asyncio.sleep(wait)
 
        if not manager.rotate():
            raise MetricRateLimited(
                f"{spec.title}: all {len(manager.keys)} Groq judge key(s) are rate-limited."
            )
        metric = spec.factory(manager.build_llm(), emb)
 
 
async def _run_metric(spec: MetricSpec, samples: list, ctx: RunContext) -> pd.DataFrame:
    """
    Generic runner used by every run_<metric>() function.
    Skips finished samples, saves after each new one, resumes where it left off.
    """
    total = len(samples)
    rows, todo = _plan(spec, samples, ctx.force)
 
    if ctx.progress_cb:
        ctx.progress_cb(spec.key, len(rows), total)
 
    if not todo:
        ctx.say(f"⏭️ {spec.title}: already complete ({total}/{total}) — loaded from checkpoint.")
        return _rows_df(spec, samples, rows)
 
    if rows:
        ctx.say(f"▶️ {spec.title}: resuming — {len(rows)}/{total} already saved, {len(todo)} to go.")
 
    if spec.uses_llm:
        ctx.manager.reset()
 
    last_error = None
 
    with logfire.span(f"🧪 Metric — {spec.title}", pending=len(todo), cached=len(rows)):
        for n, (sid, s, inp, fp) in enumerate(todo):
            if spec.uses_llm and n > 0:
                await cooldown(COOLDOWN_MINI, f"{spec.title} sample {len(rows)}", ctx.status_cb)
 
            ctx.say(f"🧪 {spec.title} — sample {len(rows) + 1}/{total}...")
 
            try:
                score = await _score_one(spec, inp, ctx) if spec.uses_llm else spec.compute(inp)
            except MetricRateLimited as exc:
                save_metric_checkpoint(spec.key, rows, str(exc))
                raise
            except Exception as exc:
                last_error = f"sample {sid}: {exc}"
                logfire.error(f"❌ {spec.title} failed on sample {sid}: {exc}")
                ctx.say(f"❌ {spec.title}: sample {sid} failed ({exc}). Continuing; it will be retried next run.")
                continue
 
            rows[sid] = {
                "question": s["question"][:120],
                "score": round(score, 4),
                "fp": fp,
                "scored_at": now_iso(),
            }
            save_metric_checkpoint(spec.key, rows, last_error)   # ← checkpoint
            if ctx.progress_cb:
                ctx.progress_cb(spec.key, len(rows), total)
 
    save_metric_checkpoint(spec.key, rows, last_error)
    return _rows_df(spec, samples, rows)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# One function per metric
# ─────────────────────────────────────────────────────────────────────────────
# Each takes already-prepared samples (see prep_samples) and a RunContext and
# returns a DataFrame [id, question, <metric>].
 
async def run_faithfulness(samples: list, ctx: RunContext) -> pd.DataFrame:
    return await _run_metric(SPECS["faithfulness"], samples, ctx)
 
 
async def run_answer_relevancy(samples: list, ctx: RunContext) -> pd.DataFrame:
    return await _run_metric(SPECS["answer_relevancy"], samples, ctx)
 
 
async def run_context_precision(samples: list, ctx: RunContext) -> pd.DataFrame:
    return await _run_metric(SPECS["context_precision"], samples, ctx)
 
 
async def run_context_recall(samples: list, ctx: RunContext) -> pd.DataFrame:
    return await _run_metric(SPECS["context_recall"], samples, ctx)
 
 
async def run_answer_correctness(samples: list, ctx: RunContext) -> pd.DataFrame:
    return await _run_metric(SPECS["answer_correctness"], samples, ctx)
 
 
async def run_tool_correctness(samples: list, ctx: RunContext) -> pd.DataFrame:
    return await _run_metric(SPECS["tool_correctness"], samples, ctx)
 
 
RUNNERS = {
    "faithfulness": run_faithfulness,
    "answer_relevancy": run_answer_relevancy,
    "context_precision": run_context_precision,
    "context_recall": run_context_recall,
    "answer_correctness": run_answer_correctness,
    "tool_correctness": run_tool_correctness,
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
 
async def run_all_metrics(
    golden_dataset: dict,
    status_cb=None,
    progress_cb=None,
    only: Optional[list] = None,
    force: bool = False,
) -> dict:
    """
    Runs every metric (or only the keys in `only`) in order.
 
    • Metrics that are already complete in the checkpoint are skipped.
    • Partially-complete metrics resume from the first unscored sample.
    • If a metric fails (e.g. rate limits), the error is reported and the NEXT
      metric still runs. Re-run later to finish the failed one.
    • force=True ignores checkpoints and recomputes everything selected.
 
    Returns {metric_key: DataFrame} for every metric that has at least one score.
    """
    samples = prep_samples(golden_dataset)
 
    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")
 
    keys = [k for k in RUN_ORDER if only is None or k in only]
    ctx = RunContext(status_cb=status_cb, progress_cb=progress_cb, force=force)
    results, failures = {}, {}
 
    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples), metrics=keys):
        for key in keys:
            spec = SPECS[key]
            _, todo = _plan(spec, samples, force)
 
            # Cooldown only when this metric will really call the LLM and the
            # previous one did too — skipped / cached metrics cost no waiting.
            if todo and spec.uses_llm and ctx.made_llm_calls:
                await cooldown(COOLDOWN_STANDARD, "previous metric", status_cb)
                ctx.made_llm_calls = False
 
            try:
                df = await RUNNERS[key](samples, ctx)
            except Exception as exc:
                failures[key] = str(exc)
                logfire.error(f"❌ {spec.title} failed: {exc}")
                ctx.say(f"❌ {spec.title} stopped: {exc} — moving on to the next metric.")
                rows = load_metric_checkpoint(key).get("rows", {})
                df = _rows_df(spec, samples, rows)
 
            if len(df):
                results[key] = df
 
        incomplete = [SPECS[k].title for k in keys
                      if len(_plan(SPECS[k], samples)[1]) > 0]
        if incomplete:
            ctx.say("⚠️ Finished with incomplete metrics: " + ", ".join(incomplete)
                    + ". Click Run again to resume them.")
        else:
            ctx.say(f"✅ All {len(keys)} selected metric(s) complete!")
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Read-only helpers for the UI (no LLM, no judge keys needed)
# ─────────────────────────────────────────────────────────────────────────────
 
def metric_status(samples: list) -> list[dict]:
    """Per-metric progress based on what is saved on disk for these samples."""
    out = []
    total = len(samples)
 
    for key in RUN_ORDER:
        spec = SPECS[key]
        valid, todo = _plan(spec, samples)
        ck = load_metric_checkpoint(key)
        done = len(valid)
 
        out.append({
            "key": key,
            "title": spec.title,
            "done": done,
            "total": total,
            "state": "complete" if not todo else ("partial" if done else "not_run"),
            "updated_at": ck.get("updated_at"),
            "last_error": ck.get("last_error"),
        })
 
    return out
 
 
def load_metric_results(samples: list) -> dict:
    """Saved scores that are still valid for these samples: {metric_key: DataFrame}."""
    results = {}
    for key in RUN_ORDER:
        spec = SPECS[key]
        valid, _ = _plan(spec, samples)
        if valid:
            results[key] = _rows_df(spec, samples, valid)
    return results
