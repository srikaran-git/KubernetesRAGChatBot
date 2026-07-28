import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
import time
import math
import logfire

BATCH_SIZE = 5          # Reduce from 50
MAX_BATCH_TOKENS = 25000
REQUEST_DELAY = 20       # seconds


_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-1, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2-preview",
                google_api_key=settings.GEMINI_API_KEY,
            )
    model.embed_query("probe")
    return model


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Approximate Gemini token count.
    Rule of thumb: 1 token ≈ 4 characters.
    """
    return math.ceil(len(text) / 4)

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Exponential backoff: 1 s → 2 s → 4 s → 8 s (4 attempts total)
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = min(60, 5 * (2 ** attempt))
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.embed_documents(batch)

# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()

    all_embeddings = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        tokens = estimate_tokens(text)

        # If adding this chunk would exceed the batch token budget,
        # send the current batch first.
        if (
            current_batch
            and (
                len(current_batch) >= BATCH_SIZE
                or current_tokens + tokens > MAX_BATCH_TOKENS
            )
        ):
            all_embeddings.extend(_embed_batch(current_batch))

            logfire.info(
                f"Embedded batch | "
                f"chunks={len(current_batch)} "
                f"tokens≈{current_tokens}"
            )

            time.sleep(REQUEST_DELAY)

            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += tokens

    # Send remaining texts
    if current_batch:
        all_embeddings.extend(_embed_batch(current_batch))

    return all_embeddings
