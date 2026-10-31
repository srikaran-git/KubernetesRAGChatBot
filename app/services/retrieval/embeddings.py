import math
import threading
import time

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

BATCH_SIZE = 5
MAX_BATCH_TOKENS = 20000
MAX_INPUT_TOKENS = 2048
# Four characters per token is only an estimate, so leave headroom below the
# provider's 2,048-token limit instead of sending oversized inputs to Gemini.
MAX_INPUT_CHARS = 7000
REQUEST_DELAY = 20        # seconds between batches

GEMINI_MODEL = "models/gemini-embedding-001"
KEY_COOLDOWN = 65         # seconds a rate-limited key is benched
MAX_FAILOVERS = 12        # max key switches for a single request before giving up

GEMINI_DIM = 1024

active_model = None

api_keys: list[str] = []
current_idx: int = 0
cooldown_until: dict[int, float] = {}   # key index -> unix time it becomes usable again
state_lock = threading.RLock()


# ── API key handling ───────────────────────────────────────────────────────────

def split_keys(value) -> list[str]:
    """Accept a string, comma-separated string, or list/tuple of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [k.strip() for k in value.split(",") if k.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(k).strip() for k in value if str(k).strip()]
    return []


def collect_api_keys() -> list[str]:
    """
    Primary key first, then every other GEMINI*KEY* setting found in config
    (e.g. GEMINI_API_KEY_2, GEMINI_FALLBACK_API_KEYS, GEMINI_API_KEYS ...).
    Duplicates are removed while keeping order.
    """
    keys: list[str] = split_keys(getattr(settings, "GEMINI_API_KEY", None))

    # dir() also finds class-level attributes (plain Settings classes)
    for name in sorted(dir(settings)):
        upper = name.upper()
        if upper == "GEMINI_API_KEY":
            continue
        if upper.startswith("GEMINI") and "KEY" in upper:
            keys.extend(split_keys(getattr(settings, name, None)))

    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def is_rate_limit(e: Exception) -> bool:
    err = str(e).lower()
    return any(x in err for x in ("429", "rate", "quota", "resource_exhausted", "resource exhausted"))


def build_model(key: str):
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_MODEL,
        google_api_key=key,
        output_dimensionality=GEMINI_DIM,
    )


def switch_key() -> None:
    """
    Move to the next usable API key. If every key is cooling down,
    sleep until the earliest one is available again.
    """
    global active_model, current_idx

    with state_lock:
        n = len(api_keys)
        now = time.time()

        # Prefer the next key in order that is not cooling down.
        for step in range(1, n + 1):
            idx = (current_idx + step) % n
            if cooldown_until.get(idx, 0) <= now:
                break
        else:
            # All keys are cooling down -> wait for the soonest one.
            idx = min(range(n), key=lambda i: cooldown_until.get(i, 0))
            wait = max(0.0, cooldown_until.get(idx, 0) - now) + 1
            logfire.warning(f"All {n} Gemini API keys are rate limited — waiting {wait:.0f}s.")
            time.sleep(wait)

        current_idx = idx
        active_model = build_model(api_keys[idx])
        logfire.info(f"Switched to Gemini API key #{idx + 1}/{n}.")


# ── Model initialisation ───────────────────────────────────────────────────────

def init():
    """Initialise the embedding model once per process. Called lazily on first use."""
    global active_model, current_idx, api_keys
    with state_lock:
        if active_model is not None:
            return

        api_keys = collect_api_keys()
        if not api_keys:
            raise RuntimeError("No Gemini API keys configured.")

        logfire.info(f"Found {len(api_keys)} Gemini API key(s).")

        last_error: Exception | None = None
        for idx, key in enumerate(api_keys):
            try:
                model = build_model(key)
                model.embed_query("probe")
                active_model = model
                current_idx = idx
                logfire.info(
                    f"Gemini embeddings ready (key #{idx + 1}, "
                    f"gemini-embedding-001, {GEMINI_DIM}-dim)."
                )
                return
            except Exception as e:
                last_error = e
                if is_rate_limit(e):
                    cooldown_until[idx] = time.time() + KEY_COOLDOWN
                logfire.warning(f"Gemini probe failed for key #{idx + 1}: {e}")

        raise RuntimeError(f"All Gemini API keys failed during initialisation: {last_error}")


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension of the embedding model."""
    return GEMINI_DIM


def estimate_tokens(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters."""
    return math.ceil(len(text) / 4)


def validate_input(text: str) -> None:
    """Reject text that cannot safely fit Gemini's per-input token limit."""
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(
            "Each text must be at most approximately "
            f"{MAX_INPUT_TOKENS} Gemini tokens ({MAX_INPUT_CHARS} characters)."
        )


# ── Calls with API key failover ────────────────────────────────────────────────

def call_with_failover(method: str, payload):
    """
    Call `embed_query` / `embed_documents` on the active Gemini model.
    On a rate-limit error, bench the current key and switch to the next one.
    """
    failovers = 0
    while True:
        with state_lock:
            model = active_model
            key_idx = current_idx
        try:
            return getattr(model, method)(payload)
        except Exception as e:
            if not is_rate_limit(e):
                logfire.error(f"Gemini embedding failed: {e}")
                raise

            with state_lock:
                cooldown_until[key_idx] = time.time() + KEY_COOLDOWN
            failovers += 1
            logfire.warning(
                f"Gemini key #{key_idx + 1} rate limited "
                f"(failover {failovers}/{MAX_FAILOVERS})."
            )

            if failovers > MAX_FAILOVERS:
                raise RuntimeError(
                    f"Gemini rate limit persisted across all API keys after {MAX_FAILOVERS} failovers."
                ) from e

            switch_key()


def embed_batch(batch: list[str]) -> list[list[float]]:
    return call_with_failover("embed_documents", batch)


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    validate_input(query)
    init()
    return call_with_failover("embed_query", query)


def embed_texts(texts: list[str]) -> list[list[float]]:
    init()

    all_embeddings = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        validate_input(text)
        tokens = estimate_tokens(text)

        if current_batch and (
            len(current_batch) >= BATCH_SIZE
            or current_tokens + tokens > MAX_BATCH_TOKENS
        ):
            all_embeddings.extend(embed_batch(current_batch))

            logfire.info(
                f"Embedded batch | chunks={len(current_batch)} tokens≈{current_tokens}"
            )

            time.sleep(REQUEST_DELAY)

            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += tokens

    if current_batch:
        all_embeddings.extend(embed_batch(current_batch))

    return all_embeddings
