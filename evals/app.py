# ─────────────────────────────────────────────────────────────────────────────
# Enterprise RAG — Evaluation Suite
# Streamlit evaluation UI
# ─────────────────────────────────────────────────────────────────────────────
 
import os
import sys
 
# logfire must be configured before application imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from dotenv import load_dotenv
load_dotenv()
 
import asyncio
import nest_asyncio
import pandas as pd
import streamlit as st
import logfire
 
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="evals",
)
 
nest_asyncio.apply()
 
from evals.pipeline import (
    run_pipeline,
    load_golden_dataset,
    load_saved_dataset,
)
from evals.metrics import (
    run_all_metrics,
    prep_samples,
    metric_status,
    load_metric_results,
    RUN_ORDER,
    SPECS,
)
from evals.checkpoint import clear_metric_checkpoint
 
 
st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)
 
 
SCORE_COLORS = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#f8d7da"}
 
# Display order = experiment numbering shown in the UI.
DISPLAY_ORDER = [
    "faithfulness", "answer_relevancy", "context_precision",
    "context_recall", "answer_correctness", "tool_correctness",
]
 
METRIC_HELP = {
    "faithfulness": ("Is every claim in the answer backed by the retrieved context?",
                     "Low score → the model is adding facts that are not in the documents (hallucination)."),
    "answer_relevancy": ("Does the answer actually address the question that was asked?",
                         "Low score → answer is off-topic, vague or padded."),
    "context_precision": ("Are the useful chunks ranked at the top of what was retrieved?",
                          "Low score → noisy or badly ordered retrieval. Look at reranking / top-k."),
    "context_recall": ("Did retrieval fetch everything needed to write the reference answer?",
                       "Low score → relevant information is missing. Look at chunking / embeddings / top-k."),
    "answer_correctness": ("How close is the answer to the reference answer (facts + meaning)?",
                           "The end-to-end quality score. Low score with good recall → generation problem."),
    "tool_correctness": ("Did the agent pick the expected tool (retrieve vs. direct answer)?",
                         "Low score → intent routing is wrong. Zero LLM calls, instant."),
}
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"
 
 
def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"
 
 
def _color_score(val):
    if not isinstance(val, (int, float)) or pd.isna(val):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}; color: #111"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}; color: #111"
    return f"background-color: {SCORE_COLORS['red']}; color: #111"
 
 
def _style_cells(df: pd.DataFrame, subset):
    """pandas >= 2.1 renamed Styler.applymap → Styler.map."""
    styler = df.style
    apply = styler.map if hasattr(styler, "map") else styler.applymap
    return apply(_color_score, subset=subset)
 
 
def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)
 
 
def _request_run(only):
    """Button callback → stores what to run; the run itself happens on the rerun."""
    if only == "selected":
        only = list(st.session_state.get("selected_metrics", []))
        if not only:
            return
    st.session_state.run_request = {
        "only": only,
        "force": bool(st.session_state.get("force_rerun", False)),
    }
 
 
def _state_label(row: dict) -> str:
    if row["state"] == "complete":
        return "✅ Complete"
    if row["state"] == "partial":
        return "🟡 Partial — will resume"
    return "⬜ Not run"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
 
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset()
 
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
 
if "enriched_dataset" not in st.session_state:
    st.session_state.enriched_dataset = None
 
if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []
 
 
golden = st.session_state.golden
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
 
st.title("🧪 Enterprise RAG — Evaluation Suite")
 
st.caption(
    "Step 1: Review ground truth → "
    "Step 2: Run live pipeline → "
    "Step 3: Score with RAGAS  ·  "
    "Progress is checkpointed to disk, so any step can be resumed."
)
 
st.divider()
 
 
tab1, tab2, tab3 = st.tabs(
    [
        "📋 Step 1 — Ground Truth",
        "🚀 Step 2 — Live Pipeline",
        "📊 Step 3 — Eval Metrics",
    ]
)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
 
with tab1:
    st.subheader("Ground Truth Dataset")
 
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your "
        "enterprise documents. Each entry has a question, a reference "
        "answer, and the expected tool the RAG agent should call."
    )
 
    rag_rows = []
 
    for s in golden["rag_samples"]:
        rag_rows.append({
            "ID": s["id"],
            "Domain": s["domain"].replace("_", " ").title(),
            "Question": s["question"],
            "Reference Answer": (
                s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"]
            ),
            "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
        })
 
    st.dataframe(pd.DataFrame(rag_rows), use_container_width=True, hide_index=True)
 
    st.caption(f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs")
 
    with st.expander("View raw golden_dataset.json"):
        st.json(golden)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
 
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
 
    st.markdown(
        "Sends each golden question to your running FastAPI app "
        "(`localhost:8000/query`). Captures the actual response, "
        "retrieved contexts, and tool called. Guardrails are not executed "
        "during this pipeline."
    )
 
    st.info(
        "⚠️ Make sure your FastAPI backend is running first: "
        "`uvicorn app.main:app --reload --port 8000`",
        icon="⚠️",
    )
 
    _, n_saved = load_saved_dataset(golden)
    n_total = len(golden["rag_samples"])
 
    if n_saved and not st.session_state.pipeline_done:
        st.success(
            f"💾 Found saved responses for **{n_saved}/{n_total}** questions. "
            + ("Load them to skip the pipeline, or run it to fill the gaps."
               if n_saved < n_total else "Load them to skip the pipeline entirely.")
        )
 
    fresh = st.checkbox(
        "Start fresh (ignore saved responses and call the API for every question)",
        key="pipeline_fresh",
        disabled=st.session_state.pipeline_done,
    )
 
    col_p1, col_p2, col_p3, col_p4 = st.columns([1.3, 1.3, 1.3, 1])
 
    run_label = "▶️ Resume Live Pipeline" if (n_saved and not fresh) else "▶️ Run Live Pipeline"
 
    run_pipeline_btn = col_p1.button(
        run_label,
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )
 
    load_saved_btn = col_p2.button(
        f"📂 Load saved ({n_saved}/{n_total})",
        width="stretch",
        disabled=st.session_state.pipeline_done or n_saved == 0,
    )
 
    reset_btn = col_p3.button(
        "🔄 Reset view",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
        help="Clears this screen only. Saved responses stay on disk and are reused on the next run.",
    )
 
    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.pipeline_rows = []
        st.rerun()
 
    if load_saved_btn:
        dataset, _ = load_saved_dataset(golden)
        st.session_state.enriched_dataset = dataset
        st.session_state.pipeline_done = True
        st.rerun()
 
    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
 
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()
 
        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
 
            if stage == "calling":
                progress_bar.progress(
                    pct, text=f"[{i+1}/{total}] Calling /query: {question[:60]}..."
                )
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
 
                st.session_state.pipeline_rows.append({
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": short_r if short_r else "⚠️ No response",
                    "Status": "✅" if short_r else "❌",
                })
 
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    use_container_width=True,
                    hide_index=True,
                )
 
                progress_bar.progress(
                    int(((i + 1) / total) * 100), text=f"[{i+1}/{total}] ✅ Done"
                )
 
        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(
                golden,
                progress_callback=pipeline_cb,
                resume=not fresh,
            )
 
            st.session_state.enriched_dataset = enriched
            st.session_state.pipeline_done = True
 
        progress_bar.progress(100, text="✅ All responses collected!")
 
        failed = [s for s in enriched["rag_samples"] if not s.get("actual_response")]
        if failed:
            status_slot.warning(
                f"💾 Saved {len(enriched['rag_samples']) - len(failed)} responses. "
                f"{len(failed)} failed — click **Reset view** then **Resume** to retry only those."
            )
        else:
            status_slot.success(
                f"💾 {len(enriched['rag_samples'])} responses saved to disk."
            )
 
    elif st.session_state.pipeline_done:
        st.success("✅ Pipeline results loaded. See below.")
 
        resp_rows = []
 
        for s in st.session_state.enriched_dataset["rag_samples"]:
            response = s.get("actual_response", "")
 
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": response[:100] + "..." if len(response) > 100 else response,
                "Tool Called": (
                    s["actual_tools_called"][0] if s.get("actual_tools_called") else "—"
                ),
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
            })
 
        st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
 
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")
 
    if not st.session_state.pipeline_done:
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        dataset = st.session_state.enriched_dataset
        samples = prep_samples(dataset)
 
        st.markdown(
            "Each metric is scored **one sample at a time and saved immediately** "
            "to `evals/results/metrics/`. Completed metrics are skipped, "
            "half-finished ones resume, and a failing metric never blocks the others."
        )
 
        st.info(
            "Judge keys: `JUDGE_GROQ_API_KEY`, `JUDGE_GROQ_API_KEY_2`, … "
            "(same numbered pattern for `GROQ_API_KEY`). Keys rotate automatically on rate limits.",
            icon="ℹ️",
        )
 
        # ── Controls ─────────────────────────────────────────────────────────
        st.checkbox(
            "Force re-run (ignore saved checkpoints for the metrics I run)",
            key="force_rerun",
        )
 
        ctl1, ctl2 = st.columns([1, 2])
 
        ctl1.button(
            "▶️ Run all missing / incomplete",
            type="primary",
            width="stretch",
            on_click=_request_run,
            args=(None,),
        )
 
        with ctl2:
            sel_col, btn_col = st.columns([3, 1])
            sel_col.multiselect(
                "Or run specific metrics",
                options=DISPLAY_ORDER,
                format_func=lambda k: SPECS[k].title,
                key="selected_metrics",
                label_visibility="collapsed",
                placeholder="Or pick specific metrics to run…",
            )
            btn_col.button(
                "▶️ Run selected",
                width="stretch",
                on_click=_request_run,
                args=("selected",),
            )
 
        # ── Run (triggered by a button callback on the previous click) ──────
        request = st.session_state.pop("run_request", None)
 
        if request:
            progress = st.progress(0, text="Starting...")
            status_slot = st.empty()
 
            def status_cb(msg: str):
                status_slot.info(msg)
 
            def progress_cb(key: str, done: int, total: int):
                progress.progress(
                    done / total if total else 0.0,
                    text=f"{SPECS[key].title} — {done}/{total} samples saved",
                )
 
            try:
                with logfire.span("📊 Streamlit — Run Metrics Button"):
                    _run_async(
                        run_all_metrics(
                            dataset,
                            status_cb=status_cb,
                            progress_cb=progress_cb,
                            only=request["only"],
                            force=request["force"],
                        )
                    )
            except Exception as exc:
                status_slot.error(f"❌ Evaluation stopped: {exc}")
            else:
                left = [r for r in metric_status(samples) if r["state"] != "complete"]
                if request["only"] is None and not left:
                    status_slot.success("✅ All 6 experiments complete!")
                elif left and request["only"] is None:
                    status_slot.warning(
                        "⚠️ Some metrics are incomplete (" + ", ".join(r["title"] for r in left)
                        + "). Click **Run all missing / incomplete** again to resume them."
                    )
                else:
                    status_slot.success("✅ Selected metrics finished.")
 
        # ── Everything below is read from disk, so it is always up to date ───
        status_rows = {r["key"]: r for r in metric_status(samples)}
        results = load_metric_results(samples)
 
        st.divider()
        st.markdown("### 🗂️ Checkpoint status")
 
        st.dataframe(
            pd.DataFrame([
                {
                    "Experiment": f"Exp {i + 1} — {SPECS[k].title}",
                    "Status": _state_label(status_rows[k]),
                    "Scored": status_rows[k]["done"] / status_rows[k]["total"]
                              if status_rows[k]["total"] else 0.0,
                    "Last saved": status_rows[k]["updated_at"] or "—",
                    "Last error": status_rows[k]["last_error"] or "",
                }
                for i, k in enumerate(DISPLAY_ORDER)
            ]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Scored": st.column_config.ProgressColumn(
                    "Scored", min_value=0.0, max_value=1.0, format="percent"
                ),
            },
        )
 
        if not results:
            st.info("No saved scores yet — click **Run all missing / incomplete** to start.")
        else:
            # ── Summary cards ────────────────────────────────────────────────
            st.markdown("### 📈 Summary")
 
            summary = []
            for k in DISPLAY_ORDER:
                if k in results:
                    avg = results[k][k].mean()
                    summary.append({
                        "key": k,
                        "Metric": SPECS[k].title,
                        "Score": float(avg),
                        "n": len(results[k]),
                        "total": status_rows[k]["total"],
                    })
 
            cols = st.columns(len(DISPLAY_ORDER))
            for col, k in zip(cols, DISPLAY_ORDER):
                row = next((r for r in summary if r["key"] == k), None)
                if row:
                    partial = "" if row["n"] == row["total"] else f" ({row['n']}/{row['total']})"
                    col.metric(
                        label=SPECS[k].title + partial,
                        value=f"{row['Score']:.2f}",
                        delta=_grade(row["Score"]),
                        delta_color="off",
                    )
                else:
                    col.metric(label=SPECS[k].title, value="—", delta="not run", delta_color="off")
 
            chart_df = pd.DataFrame(summary).set_index("Metric")[["Score"]]
            st.bar_chart(chart_df, y="Score", height=260)
 
            # ── One tab per metric ───────────────────────────────────────────
            st.markdown("### 🔬 Per-metric results")
 
            metric_tabs = st.tabs(
                [f"{_badge(next(r['Score'] for r in summary if r['key'] == k))} {SPECS[k].title}"
                 if k in results else f"⬜ {SPECS[k].title}"
                 for k in DISPLAY_ORDER]
            )
 
            for tab, k in zip(metric_tabs, DISPLAY_ORDER):
                with tab:
                    what, hint = METRIC_HELP[k]
                    st.markdown(f"**{what}**")
                    st.caption(hint)
 
                    if k not in results:
                        st.info("Not scored yet.")
                    else:
                        df = results[k]
                        s_ = df[k]
 
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Average", f"{s_.mean():.2f}", _grade(s_.mean()), delta_color="off")
                        m2.metric("Lowest", f"{s_.min():.2f}")
                        m3.metric("Highest", f"{s_.max():.2f}")
                        m4.metric("Below 0.5", f"{int((s_ < 0.5).sum())} / {len(s_)}")
 
                        st.dataframe(
                            _style_cells(df[["question", k]], [k]).format({k: "{:.3f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
 
                    st.button(
                        f"▶️ {'Re-run' if k in results and status_rows[k]['state'] == 'complete' else 'Run'} "
                        f"{SPECS[k].title}",
                        key=f"run_{k}",
                        on_click=_request_run,
                        args=([k],),
                        help="Uses the 'Force re-run' checkbox above to decide whether to ignore saved scores.",
                    )
 
            # ── Combined view ────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🧮 All metrics per question")
 
            matrix = None
            for k in DISPLAY_ORDER:
                if k in results:
                    part = results[k][["id", "question", k]]
                    matrix = part if matrix is None else matrix.merge(
                        part.drop(columns=["question"]), on="id", how="outer"
                    )
 
            score_cols = [k for k in DISPLAY_ORDER if k in results]
            matrix["average"] = matrix[score_cols].mean(axis=1).round(3)
            matrix = matrix.rename(columns={k: SPECS[k].title for k in score_cols})
            shown = [SPECS[k].title for k in score_cols] + ["average"]
 
            st.dataframe(
                _style_cells(matrix, shown).format({c: "{:.2f}" for c in shown}, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )
 
            worst = matrix.sort_values("average").head(5)[["id", "question", "average"]]
            with st.expander("🚨 5 weakest questions (lowest average across metrics)"):
                st.dataframe(
                    _style_cells(worst, ["average"]).format({"average": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
 
            d1, d2, _ = st.columns([1, 1, 3])
            d1.download_button(
                "⬇️ Download CSV",
                matrix.to_csv(index=False).encode("utf-8"),
                file_name="rag_eval_scores.csv",
                mime="text/csv",
                width="stretch",
            )
            d2.download_button(
                "⬇️ Download JSON",
                matrix.to_json(orient="records", indent=2).encode("utf-8"),
                file_name="rag_eval_scores.json",
                mime="application/json",
                width="stretch",
            )
 
        # ── Danger zone ──────────────────────────────────────────────────────
        with st.expander("🗑️ Clear saved metric scores"):
            st.caption("Deletes the checkpoint files so the next run recomputes everything.")
            c1, c2 = st.columns([2, 1])
            to_clear = c1.selectbox(
                "Metric", ["All metrics"] + DISPLAY_ORDER,
                format_func=lambda k: k if k == "All metrics" else SPECS[k].title,
                label_visibility="collapsed",
            )
            if c2.button("Clear", width="stretch"):
                clear_metric_checkpoint(None if to_clear == "All metrics" else to_clear)
                st.rerun()
