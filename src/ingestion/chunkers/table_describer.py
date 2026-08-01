import hashlib
import json
from pathlib import Path

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_DESCRIBE_PROMPT = """\
Convert this financial table into a concise natural-language paragraph (3-6 sentences) \
stating its key figures, so it reads as clear prose rather than a raw table grid. \
Preserve exact numbers, labels, and units. Do not add any information not present in \
the table, and do not add commentary or opinions.

Section: {section}
Caption: {caption}

Table:
{table_markdown}

Paragraph:
"""

_CACHE_PATH = Path("data/.cache/table_description_cache.jsonl")


def _cache_key(table_markdown: str, caption: str, section: str) -> str:
    return hashlib.sha256(f"{section}:{caption}:{table_markdown}".encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, str]:
    if not _CACHE_PATH.exists():
        return {}
    cache = {}
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[entry["hash"]] = entry["description"]
    return cache


def _append_cache(key: str, description: str) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"hash": key, "description": description}) + "\n")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_claude(client: anthropic.Anthropic, model: str, section: str, caption: str, table_markdown: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": _DESCRIBE_PROMPT.format(section=section, caption=caption, table_markdown=table_markdown[:3000]),
        }],
    )
    return response.content[0].text.strip()


def describe_table(section: str, caption: str, table_markdown: str, api_key: str, model: str = "claude-haiku-4-5") -> str:
    """Convert a table into a natural-language paragraph for embedding/reranking purposes,
    since cross-encoder rerankers systematically score raw markdown-table syntax below
    prose regardless of relevance (see D-16). Retries transient failures; on final
    failure, falls back to an empty string rather than aborting ingestion. Cached on disk
    keyed by content hash, so re-ingesting unchanged tables skips the paid LLM call."""
    key = _cache_key(table_markdown, caption, section)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    client = anthropic.Anthropic(api_key=api_key)
    try:
        description = _call_claude(client, model, section, caption, table_markdown)
    except Exception:
        return ""  # fail open — ship the table without a prose description

    _append_cache(key, description)
    return description
