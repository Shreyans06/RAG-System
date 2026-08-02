import json
import time
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from src.config import get_settings
from src.ingestion.chunkers._tokenizer import get_tokenizer
from src.ingestion.models import Chunk, ContentType
from src.models.factory import get_sparse_embeddings
from src.retrieval.vector_store import upsert_chunks

_JOBS_DIR = Path("data/.cache/batch_jobs")
_DONE_DIR = _JOBS_DIR / "done"
_TERMINAL_STATUSES = ("completed", "failed", "expired", "cancelled")
_MAX_TOKENS_PER_BATCH = 2_500_000  # conservative margin under OpenAI's 3M enqueued-token cap


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().OPENAI_API_KEY)


def _chunks_path(label: str) -> Path:
    return _JOBS_DIR / f"{label}.chunks.json"


def _job_path(label: str) -> Path:
    return _JOBS_DIR / f"{label}.job.json"


def _load_chunks(label: str) -> list[Chunk]:
    raw = json.loads(_chunks_path(label).read_text())
    return [Chunk(**{**c, "content_type": ContentType(c["content_type"])}) for c in raw]


def queue_for_batch(label: str, chunks: list[Chunk]) -> int:
    """Stages chunks on disk for later batch submission — does NOT call the OpenAI API.
    submit_queued_batches() handles actually submitting, one batch at a time (see its
    docstring for why). Deduplicates by chunk ID first: SEC filing cover pages carry small
    boilerplate tables (checkbox/signature tables, "Securities registered pursuant to...")
    that are byte-identical across many of a company's filings, and IDs are content-hash
    based, so these collide across filings. Harmless for a normal upsert (same ID -> same
    point, just overwritten), but the Batch API hard-rejects a whole batch over any
    duplicate custom_id — so queue each unique chunk once.

    Also splits into token-budgeted sub-batches (labeled "{label}-1", "{label}-2", ...)
    when a single ticker's chunks alone would exceed OpenAI's enqueued-token cap — confirmed
    live: one company's filings (12,147 chunks, ~3.1M tokens) exceeded the 3M limit on their
    own, independent of anything else queued. Returns the deduplicated chunk count."""
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    deduped = list({c.id: c for c in chunks}.values())
    if len(deduped) != len(chunks):
        print(f"  ({len(chunks) - len(deduped)} duplicate chunk ID(s) collapsed)")

    tokenizer = get_tokenizer()
    sub_batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_tokens = 0
    for chunk in deduped:
        n = len(tokenizer.encode(chunk.content))
        if current and current_tokens + n > _MAX_TOKENS_PER_BATCH:
            sub_batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += n
    if current:
        sub_batches.append(current)

    for i, batch_chunks in enumerate(sub_batches, start=1):
        sub_label = label if len(sub_batches) == 1 else f"{label}-{i}"
        _chunks_path(sub_label).write_text(json.dumps([asdict(c) for c in batch_chunks]))
    if len(sub_batches) > 1:
        print(f"  (split into {len(sub_batches)} sub-batches to stay under the enqueued-token cap)")

    return len(deduped)


def list_pending_jobs() -> list[dict]:
    if not _JOBS_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(_JOBS_DIR.glob("*.job.json"))]


def list_queued_labels() -> list[str]:
    """Tickers that have staged chunks but haven't been submitted to OpenAI yet."""
    if not _JOBS_DIR.exists():
        return []
    submitted = {j["label"] for j in list_pending_jobs()}
    labels = [p.name.removesuffix(".chunks.json") for p in sorted(_JOBS_DIR.glob("*.chunks.json"))]
    return [label for label in labels if label not in submitted]


def _has_active_job() -> bool:
    client = _client()
    for job in list_pending_jobs():
        batch = client.batches.retrieve(job["batch_id"])
        if batch.status not in _TERMINAL_STATUSES:
            return True
    return False


def submit_queued_batches() -> str | None:
    """Submits the next queued (staged-but-unsubmitted) ticker as an OpenAI Batch API job,
    but only if no other batch is currently in flight. OpenAI caps *enqueued* tokens per
    embedding model across all non-completed batches at once (3M for
    text-embedding-3-large) — submitting a second large batch while one is still running
    can blow through that ceiling and get the whole thing rejected before processing a
    single request (confirmed live). Keeping strictly one batch in flight at a time avoids
    that regardless of how large any individual ticker's batch is. Returns the label
    submitted, or None if nothing was submitted."""
    if _has_active_job():
        return None

    queued = list_queued_labels()
    if not queued:
        return None

    label = queued[0]
    chunks = _load_chunks(label)
    settings = get_settings()
    model = settings.retrieval.get("embedding_model", "text-embedding-3-large")

    request_path = _JOBS_DIR / f"{label}.requests.jsonl"
    with request_path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "custom_id": chunk.id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": model, "input": chunk.content},
            }) + "\n")

    client = _client()
    uploaded = client.files.create(file=request_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={"label": label},
    )
    request_path.unlink()

    _job_path(label).write_text(json.dumps({
        "batch_id": batch.id,
        "label": label,
        "num_chunks": len(chunks),
        "submitted_at": time.time(),
    }))
    return label


def complete_batch_job(job: dict, qdrant: QdrantClient, collection_name: str) -> str:
    """Checks one submitted batch job's status. If OpenAI has finished it, downloads the
    embeddings, matches them back to their chunks by custom_id, generates sparse vectors
    locally (unaffected by the batch API — FastEmbed runs on-machine, no API cost either
    way), and upserts everything to Qdrant. If the batch failed/expired/cancelled (e.g. the
    enqueued-token-limit rejection), drops the stale job record so the ticker's
    already-staged chunks go back to "queued" and get retried by the next
    submit_queued_batches() call, rather than being stuck forever. Returns a status string."""
    client = _client()
    batch = client.batches.retrieve(job["batch_id"])

    if batch.status in ("failed", "expired", "cancelled"):
        _job_path(job["label"]).unlink()
        return f"{batch.status} — re-queued for retry"

    if batch.status != "completed":
        return batch.status

    chunks = _load_chunks(job["label"])
    chunks_by_id = {c.id: c for c in chunks}

    embeddings_by_id: dict[str, list[float]] = {}
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = row.get("response", {}).get("body", {})
            data = body.get("data")
            if data:
                embeddings_by_id[row["custom_id"]] = data[0]["embedding"]

    failed_ids = set(chunks_by_id) - set(embeddings_by_id)
    if failed_ids and batch.error_file_id:
        errors_text = client.files.content(batch.error_file_id).text
        print(f"  {len(failed_ids)} chunk(s) failed within the batch:\n{errors_text[:500]}")

    ready_chunks = [chunks_by_id[cid] for cid in chunks_by_id if cid in embeddings_by_id]
    if ready_chunks:
        dense_embeddings = [embeddings_by_id[c.id] for c in ready_chunks]
        sparse_raw = get_sparse_embeddings().embed_documents([c.content for c in ready_chunks])
        sparse_embeddings = [SparseVector(indices=v.indices, values=v.values) for v in sparse_raw]
        upsert_chunks(qdrant, collection_name, ready_chunks, dense_embeddings, sparse_embeddings=sparse_embeddings)

    _DONE_DIR.mkdir(parents=True, exist_ok=True)
    _chunks_path(job["label"]).rename(_DONE_DIR / _chunks_path(job["label"]).name)
    _job_path(job["label"]).rename(_DONE_DIR / _job_path(job["label"]).name)

    return f"completed — {len(ready_chunks)} chunks upserted, {len(failed_ids)} failed"
