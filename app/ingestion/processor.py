"""Universal document ingestion for the Enterprise Hybrid RAG API.

Pipeline:
    Parse -> Chunk -> Dense + Sparse embeddings -> Qdrant hybrid indexing
    -> Save checkpoint

The existing dense-only collection is preserved. By default, documents are
indexed into `enterprise_rag_hybrid`.

Usage:
    python -m app.ingestion.processor DATA
    python -m app.ingestion.processor DATA/true_data true
    python -m app.ingestion.processor DATA --wipe
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import logfire
from qdrant_client import QdrantClient, models

from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embeddings import GEMINI_MODEL, get_embedding_dim
from app.services.retrieval.hybrid_index import (
    ensure_hybrid_collection,
    index_chunks,
)

logfire.configure(service_name="Enterprise-Ingestion-Service")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = Path("processed_data")

# Keep the original dense-only collection intact.
# You can override this with QDRANT_HYBRID_COLLECTION in your environment.
HYBRID_COLLECTION = (
    getattr(settings, "QDRANT_HYBRID_COLLECTION", None)
    or os.getenv("QDRANT_HYBRID_COLLECTION")
    or "enterprise_rag_hybrid"
)

DEFAULT_BATCH_SIZE = 20
EMBEDDING_DIMENSION = get_embedding_dim()
EMBEDDING_VERSION = f"{GEMINI_MODEL.removeprefix('models/')}-{EMBEDDING_DIMENSION}d"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".docx",
    ".pptx",
}


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


# ---------------------------------------------------------------------------
# Local persistence and checkpoint helpers
# ---------------------------------------------------------------------------

def safe_filename(filename: str) -> str:
    """Convert a filename into a filesystem-safe label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", filename)


def source_key(source_type: str, filename: str) -> str:
    """Build a unique source identifier, including the source category."""
    return f"{source_type.strip()}/{filename.strip()}"


def infer_source_type(path_name: str, explicit_source_type: Optional[str]) -> str:
    """Use an explicit category when supplied, otherwise infer a stable one."""
    if explicit_source_type:
        return explicit_source_type

    name = path_name.lower()
    if "true" in name:
        return "true"
    if "noisy" in name:
        return "noisy"
    return "general"


def content_hash(chunks: list[str]) -> str:
    """Hash chunk content so stale checkpoints are not trusted."""
    serialized = json.dumps(
        chunks,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def checkpoint_path(source_key: str) -> Path:
    """Return a stable checkpoint path without filename collisions."""
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
    label = safe_filename(Path(source_key).name)
    checkpoint_dir = PROCESSED_DATA_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir / f"{label}_{digest}.checkpoint.json"


def save_json_atomically(path: Path, data: dict) -> None:
    """Write JSON through a temporary file to avoid partial checkpoint files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def load_checkpoint(path: Path) -> Optional[dict]:
    """Read a checkpoint, returning None if it is missing or invalid."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            checkpoint = json.load(file)

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a JSON object")

        return checkpoint

    except (OSError, json.JSONDecodeError, ValueError) as error:
        logfire.warning(
            "Checkpoint could not be read; the document will be reindexed.",
            path=str(path),
            error=str(error),
        )
        return None


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

def extract_text(file_path: str, filename: str) -> str:
    """Parse a supported document into plain text."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return parse_pdf(file_path)

    if extension in {".html", ".htm"}:
        return parse_html(file_path)

    if extension == ".txt":
        return parse_text(file_path)

    if extension in {".docx", ".pptx"}:
        from app.ingestion.loaders.office import parse_office

        return parse_office(file_path)

    raise ValueError(f"Unsupported file extension: {extension}")


# ---------------------------------------------------------------------------
# Hybrid collection setup
# ---------------------------------------------------------------------------

def ensure_ingestion_collection(wipe: bool = False) -> None:
    """Ensure the hybrid collection exists without modifying the old collection.

    When wipe=True, only the hybrid collection is deleted and recreated.
    The original settings.QDRANT_COLLECTION is never deleted here.
    """
    if wipe and qdrant_client.collection_exists(HYBRID_COLLECTION):
        logfire.warning(
            "Deleting the hybrid collection because --wipe was requested.",
            collection=HYBRID_COLLECTION,
        )
        qdrant_client.delete_collection(
            collection_name=HYBRID_COLLECTION,
        )

    ensure_hybrid_collection(
        client=qdrant_client,
        collection_name=HYBRID_COLLECTION,
        dense_dim=EMBEDDING_DIMENSION,
    )

    logfire.info(
        "Hybrid Qdrant collection is ready.",
        collection=HYBRID_COLLECTION,
        dense_dimension=EMBEDDING_DIMENSION,
    )


def source_is_indexed(source: str, expected_chunks: int) -> bool:
    """Confirm the current collection still contains every chunk for a source."""
    result = qdrant_client.count(
        collection_name=HYBRID_COLLECTION,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source),
                )
            ]
        ),
        exact=True,
    )
    return result.count == expected_chunks


def delete_missing_sources(source_scope: str, current_sources: set[str]) -> int:
    """Remove vectors for files no longer present in a fully scanned scope."""
    deleted = 0
    offset = None
    source_scope_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_scope",
                match=models.MatchValue(value=source_scope),
            )
        ]
    )

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=HYBRID_COLLECTION,
            scroll_filter=source_scope_filter,
            offset=offset,
            limit=100,
            with_payload=["source"],
            with_vectors=False,
        )
        stale_ids = [
            point.id
            for point in points
            if (point.payload or {}).get("source") not in current_sources
        ]
        if stale_ids:
            qdrant_client.delete(
                collection_name=HYBRID_COLLECTION,
                points_selector=stale_ids,
                wait=True,
            )
            deleted += len(stale_ids)

        if offset is None:
            return deleted


# ---------------------------------------------------------------------------
# Single-file ingestion
# ---------------------------------------------------------------------------

def process_file(
    file_path: str,
    filename: str,
    source_type: str,
    source_scope: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_reindex: bool = False,
) -> bool:
    """Parse and index a file into the hybrid collection.

    Returns:
        True if the file was indexed or was already up to date.
        False if the file was unsupported or contained no usable text.

    Exceptions during parsing, embedding, or indexing are logged and raised
    again so the caller cannot mistakenly report a successful ingestion.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        logfire.warning(
            "Skipping unsupported file type.",
            filename=filename,
            extension=extension,
        )
        return False

    source_scope = source_scope or source_type
    source_id = source_key(source_scope, filename)
    checkpoint_file = checkpoint_path(source_id)

    with logfire.span(
        "Processing File",
        file=filename,
        source=source_type,
        collection=HYBRID_COLLECTION,
    ):
        try:
            # 1. Extract text.
            full_text = extract_text(file_path, filename)

            if not full_text or not full_text.strip():
                logfire.warning(
                    "No text extracted; skipping document.",
                    filename=filename,
                )
                return False

            # 2. Create clean chunks. Filtering them here keeps chunk indices
            #    consistent with the hybrid indexer's deterministic IDs.
            chunks = [
                chunk.strip()
                for chunk in chunk_text(full_text)
                if chunk and chunk.strip()
            ]

            if not chunks:
                logfire.warning(
                    "No usable chunks were generated.",
                    filename=filename,
                )
                return False

            total_chunks = len(chunks)
            content_digest = content_hash(chunks)

            # 3. Skip only when the checkpoint confirms that the same content
            #    was fully indexed. Older checkpoints without a content hash
            #    will trigger one safe reindex.
            checkpoint = load_checkpoint(checkpoint_file)

            if (
                not force_reindex
                and checkpoint is not None
                and checkpoint.get("completed") is True
                and checkpoint.get("content_hash") == content_digest
                and checkpoint.get("total_chunks") == total_chunks
                and checkpoint.get("source") == source_id
                and checkpoint.get("collection") == HYBRID_COLLECTION
                and checkpoint.get("embedding_version") == EMBEDDING_VERSION
                and checkpoint.get("embedding_dimension") == EMBEDDING_DIMENSION
                and source_is_indexed(source_id, total_chunks)
            ):
                logfire.info(
                    "Document is already up to date.",
                    filename=filename,
                    chunk_count=total_chunks,
                )
                return True

            # 4. Index the complete file.
            #
            # index_chunks() performs its own batched embedding/upsert work.
            # Calling it once for the full document is important: its cleanup
            # of stale tail chunks must not run after an individual batch.
            logfire.info(
                "Starting dense + sparse indexing.",
                filename=filename,
                chunks=total_chunks,
                batch_size=batch_size,
                collection=HYBRID_COLLECTION,
            )

            indexed_count = index_chunks(
                client=qdrant_client,
                collection_name=HYBRID_COLLECTION,
                chunks=chunks,
                source=source_id,
                source_type=source_type,
                source_scope=source_scope,
                batch_size=batch_size,
            )

            if indexed_count != total_chunks:
                raise RuntimeError(
                    f"Expected to index {total_chunks} chunks for "
                    f"{filename}, but index_chunks reported {indexed_count}."
                )

            # 5. Save the completion checkpoint only after indexing succeeds.
            completed_checkpoint = {
                "filename": filename,
                "source_type": source_type,
                "source_scope": source_scope,
                "source": source_id,
                "content_hash": content_digest,
                "total_chunks": total_chunks,
                "next_chunk_index": total_chunks,
                "completed": True,
                "collection": HYBRID_COLLECTION,
                "embedding_version": EMBEDDING_VERSION,
                "embedding_dimension": EMBEDDING_DIMENSION,
            }

            save_json_atomically(
                checkpoint_file,
                completed_checkpoint,
            )

            logfire.info(
                "Document indexed successfully.",
                filename=filename,
                indexed_chunks=indexed_count,
                collection=HYBRID_COLLECTION,
                checkpoint=str(checkpoint_file),
            )

            return True

        except Exception as error:
            logfire.error(
                "Document ingestion failed.",
                filename=filename,
                source=source_type,
                collection=HYBRID_COLLECTION,
                error=str(error),
            )
            raise


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------

def process_directory(
    dir_path: str,
    source_type: str,
    force_reindex: bool = False,
    recursive: bool = True,
    source_scope: Optional[str] = None,
) -> None:
    """Process every supported file in a directory.

    All files are attempted. If one or more fail, a RuntimeError is raised
    at the end so automated jobs can detect the failure.
    """
    directory = Path(dir_path)
    source_scope = source_scope or source_type

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    with logfire.span(
        "Scanning Directory",
        path=str(directory),
        source=source_type,
    ):
        candidates = directory.rglob("*") if recursive else directory.iterdir()
        files = sorted(path for path in candidates if path.is_file())

        logfire.info(
            "Directory scan completed.",
            directory=str(directory),
            file_count=len(files),
        )

        failures: list[str] = []
        succeeded = 0
        skipped = 0
        current_sources = {
            source_key(
                source_scope,
                file_path.relative_to(directory).as_posix(),
            )
            for file_path in files
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        for file_path in files:
            try:
                processed = process_file(
                    file_path=str(file_path),
                    filename=file_path.relative_to(directory).as_posix(),
                    source_type=source_type,
                    source_scope=source_scope,
                    force_reindex=force_reindex,
                )

                if processed:
                    succeeded += 1
                else:
                    skipped += 1

            except Exception as error:
                failures.append(f"{file_path.name}: {error}")
                logfire.error(
                    "File failed; continuing with the next file.",
                    filename=file_path.name,
                    error=str(error),
                )

        logfire.info(
            "Directory ingestion finished.",
            directory=str(directory),
            succeeded=succeeded,
            skipped=skipped,
            failed=len(failures),
        )

        if failures:
            details = "\n".join(f"- {failure}" for failure in failures)
            raise RuntimeError(
                f"{len(failures)} file(s) failed in {directory}:\n{details}"
            )

        deleted = delete_missing_sources(source_scope, current_sources)
        if deleted:
            logfire.info(
                "Removed vectors for deleted source files.",
                source_scope=source_scope,
                point_count=deleted,
            )


# ---------------------------------------------------------------------------
# Universal ingestion entry point
# ---------------------------------------------------------------------------

def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: Optional[str] = None,
    wipe: bool = False,
) -> None:
    """Ingest a directory tree, mapping subdirectories to source types.

    If the input directory has subdirectories, each direct subdirectory is
    treated as a source category. Otherwise, explicit_source_type is used,
    falling back to a category inferred from the directory name.

    --wipe deletes only HYBRID_COLLECTION and forces files to be reindexed.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {base_dir}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {base_dir}")

    with logfire.span(
        "Universal Ingestion Started",
        base_directory=str(base_path),
        collection=HYBRID_COLLECTION,
    ):
        # Create the hybrid collection before processing any files.
        ensure_ingestion_collection(wipe=wipe)

        subdirectories = sorted(
            path
            for path in base_path.iterdir()
            if path.is_dir()
        )

        if not subdirectories:
            source_type = infer_source_type(base_path.name, explicit_source_type)

            logfire.info(
                "Processing input directory as a single source.",
                directory=str(base_path),
                source_type=source_type,
            )

            process_directory(
                dir_path=str(base_path),
                source_type=source_type,
                force_reindex=wipe,
            )

        else:
            failures: list[str] = []

            # A mixed layout may contain files at the root as well as category
            # subdirectories. Process the root files without traversing into the
            # categories, which are handled below with their own source types.
            if any(path.is_file() for path in base_path.iterdir()):
                root_source_type = infer_source_type(
                    base_path.name,
                    explicit_source_type,
                )
                try:
                    process_directory(
                        dir_path=str(base_path),
                        source_type=root_source_type,
                        force_reindex=wipe,
                        recursive=False,
                        source_scope=f"{root_source_type}:root",
                    )
                except Exception as error:
                    failures.append(f"{base_path.name}: {error}")
                    logfire.error(
                        "Root-level files failed.",
                        directory=str(base_path),
                        error=str(error),
                    )

            for subdirectory in subdirectories:
                source_type = infer_source_type(
                    subdirectory.name,
                    explicit_source_type,
                )

                try:
                    process_directory(
                        dir_path=str(subdirectory),
                        source_type=source_type,
                        force_reindex=wipe,
                        source_scope=(
                            f"{source_type}:"
                            f"{subdirectory.relative_to(base_path).as_posix()}"
                        ),
                    )
                except Exception as error:
                    failures.append(f"{subdirectory.name}: {error}")
                    logfire.error(
                        "Subdirectory ingestion failed.",
                        directory=str(subdirectory),
                        error=str(error),
                    )

            if failures:
                details = "\n".join(f"- {failure}" for failure in failures)
                raise RuntimeError(
                    f"Ingestion failed for {len(failures)} "
                    f"subdirectory/subdirectories:\n{details}"
                )

        logfire.info(
            "Universal ingestion completed successfully.",
            directory=str(base_path),
            collection=HYBRID_COLLECTION,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Command-line entry point."""
    args = sys.argv[1:]
    wipe_requested = "--wipe" in args
    args = [arg for arg in args if arg != "--wipe"]

    target_dir = args[0] if args else "DATA"
    explicit_type = args[1] if len(args) > 1 else None

    if not Path(target_dir).exists():
        print(f"Error: path '{target_dir}' does not exist.")
        return 1

    try:
        run_universal_ingestion(
            base_dir=target_dir,
            explicit_source_type=explicit_type,
            wipe=wipe_requested,
        )
    except Exception as error:
        logfire.error(
            "Ingestion job failed.",
            directory=target_dir,
            error=str(error),
        )
        print(f"Ingestion failed: {error}")
        return 1

    print(
        "Ingestion completed successfully. "
        f"Hybrid collection: {HYBRID_COLLECTION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
