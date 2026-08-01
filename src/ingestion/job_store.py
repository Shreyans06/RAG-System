import json
import uuid
from datetime import UTC, datetime

import redis

from src.config import get_settings

_JOB_TTL_SECONDS = 3600

def _get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.REDIS_URL)

def _now() -> str:
    return datetime.now(UTC).isoformat()

def create_job(ticker: str, filing_types: list[str], years: list[int]) -> str:
    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id": job_id,
        "ticker": ticker.upper(),
        "filing_types": filing_types,
        "years": years,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "result": None,
        "error": None,
    }
    r = _get_redis()
    r.set(f"ingest_job:{job_id}", json.dumps(job), ex=_JOB_TTL_SECONDS)
    r.set(f"ingest_job_latest:{ticker.upper()}", job_id, ex=_JOB_TTL_SECONDS)
    return job_id

def update_job(job_id: str, **fields) -> None:
    r = _get_redis()
    raw = r.get(f"ingest_job:{job_id}")
    if raw is None:
        return
    job = json.loads(raw)
    job.update(fields)
    r.set(f"ingest_job:{job_id}", json.dumps(job), ex=_JOB_TTL_SECONDS)


def get_job(job_id: str) -> dict | None:
    r = _get_redis()
    raw = r.get(f"ingest_job:{job_id}")
    return json.loads(raw) if raw else None


def get_latest_job_for_ticker(ticker: str) -> dict | None:
    r = _get_redis()
    job_id = r.get(f"ingest_job_latest:{ticker.upper()}")
    if job_id is None:
        return None
    job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
    return get_job(job_id)