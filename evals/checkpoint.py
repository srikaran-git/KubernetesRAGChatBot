"""
Checkpoint storage for the evaluation suite.
 
Everything is stored as small JSON files under  evals/results/  (override with
the EVAL_RESULTS_DIR env var):
 
    results/pipeline_results.json      live responses from /query  (Phase 1)
    results/metrics/<metric>.json      per-sample scores           (Phase 2)
 
Metric rows carry a fingerprint of the exact inputs that produced them, so a
saved score is only reused when the question / response / contexts / reference
/ judge model are unchanged. Re-run the pipeline with new answers and the stale
scores are re-computed automatically; untouched ones are reused.
"""
 
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
 
RESULTS_DIR = Path(
    os.getenv("EVAL_RESULTS_DIR") or (Path(__file__).resolve().parent / "results")
)
METRICS_DIR = RESULTS_DIR / "metrics"
PIPELINE_FILE = RESULTS_DIR / "pipeline_results.json"
 
 
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
 
 
def fingerprint(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
 
 
def _write_json_atomic(path: Path, data) -> None:
    """Write via temp file + rename so a crash can never leave a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
 
 
def _read_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
 
 
# ── Metric checkpoints ───────────────────────────────────────────────────────
 
def load_metric_checkpoint(key: str) -> dict:
    return _read_json(METRICS_DIR / f"{key}.json", {"rows": {}})
 
 
def save_metric_checkpoint(key: str, rows: dict, last_error: str | None = None) -> None:
    _write_json_atomic(
        METRICS_DIR / f"{key}.json",
        {"metric": key, "updated_at": now_iso(), "last_error": last_error, "rows": rows},
    )
 
 
def clear_metric_checkpoint(key: str | None = None) -> None:
    """Delete one metric's checkpoint, or all of them when key is None."""
    paths = [METRICS_DIR / f"{key}.json"] if key else list(METRICS_DIR.glob("*.json"))
    for p in paths:
        if p.exists():
            p.unlink()
 
 
# ── Pipeline checkpoints ─────────────────────────────────────────────────────
 
def load_pipeline_samples() -> dict:
    return _read_json(PIPELINE_FILE, {}).get("samples", {})
 
 
def save_pipeline_sample(sample_id, data: dict) -> None:
    samples = load_pipeline_samples()
    samples[str(sample_id)] = {**data, "saved_at": now_iso()}
    _write_json_atomic(PIPELINE_FILE, {"samples": samples})
 
 
def clear_pipeline_samples() -> None:
    if PIPELINE_FILE.exists():
        PIPELINE_FILE.unlink()
 