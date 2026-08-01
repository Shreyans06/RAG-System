import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_LOG_PATH = Path("logs/queries.jsonl")


def log_query_event(
    method: str,
    question: str,
    answer: str,
    session_id: str | None,
    filters: dict[str, Any] | None,
    retrieved_chunk_ids: list[str],
    citation_metrics: dict[str, Any],
    latency_ms: float,
) -> None:
    """Append a structured JSON record of one query to logs/queries.jsonl."""
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": method,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "filters": filters,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "latency_ms": round(latency_ms, 2),
        **citation_metrics,
    }
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        logger.warning("Failed to write query log event", exc_info=True)
