import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.chunkers._tokenizer import get_tokenizer, split_into_token_windows
from src.ingestion.models import Chunk, IngestionItem

_CONTEXT_PROMPT = """\
Here is a document excerpt. Write ONE short sentence (max 20 words) that
situates this excerpt in the overall document for retrieval purposes.
Do NOT summarize - give context only.

<excerpt>{chunk_text}</excerpt>

Context sentence:
"""

_CACHE_PATH = Path("data/.cache/context_cache.jsonl")
_DEFAULT_MAX_WORKERS = 5 

def _cache_key(chunk_text: str) -> str:
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

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
            cache[entry["hash"]] = entry["context"]
    return cache


def _append_cache(key: str, context: str) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"hash": key, "context": context}) + "\n")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1 , min=2 , max=10))
def _call_claude(client: anthropic.Anthropic, model: str, chunk_text: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=60,
        messages=[{"role" : "user" , "content" : _CONTEXT_PROMPT.format(chunk_text = chunk_text[:800])}],
    )
    return response.content[0].text.strip()
    

def chunk_with_context(
    doc: IngestionItem,
    api_key: str,
    model: str = "claude-haiku-4-5",
    chunk_size: int = 400,
    overlap: int = 40,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[Chunk]:
    """
    Chunk a document into smaller chunks and generate context sentences for each chunk.
    Retries transient API failures; on final failure, falls back to no context sentence
    rather than aborting the whole ingestion. Caches context sentences on disk keyed by
    content hash, so re-ingesting unchanged boilerplate skips the paid LLM call entirely.

    Cache misses are resolved concurrently via a thread pool (bounded by max_workers) —
    the Anthropic SDK calls are I/O-bound, so threads give a real speedup without needing
    to make the surrounding (synchronous) ingestion pipeline async. All cache writes happen
    in the main thread while collecting results, never inside a worker thread, so there's no
    concurrent-file-write race to guard against.
    """
    chunks = []
    tokenizer = get_tokenizer()
    client = anthropic.Anthropic(api_key=api_key)
    cache = _load_cache()

    chunk_texts = list(split_into_token_windows(doc.content, chunk_size, overlap))

    contexts: list[str] = [""] * len(chunk_texts)
    misses: list[int] = []
    for i, text in enumerate(chunk_texts):
        key = _cache_key(text)
        if key in cache:
            contexts[i] = cache[key]
        else:
            misses.append(i)

    if misses:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_call_claude, client, model, chunk_texts[i]): i
                for i in misses
            }
            for future, i in future_to_idx.items():
                try:
                    context = future.result()
                except Exception:
                    context = ""  # fail open — ship this chunk without a context sentence
                else:
                    key = _cache_key(chunk_texts[i])
                    cache[key] = context
                    _append_cache(key, context)
                contexts[i] = context

    for idx, (chunk_text, context) in enumerate(zip(chunk_texts, contexts)):
        enriched_text = f"{context}\n\n{chunk_text}" if context else chunk_text

        chunks.append(Chunk(
            id=f"{doc.id}-ctx{idx}",
            document_id=doc.id,
            parent_chunk_id=None,
            content=enriched_text,
            content_type=doc.content_type,
            token_count=len(tokenizer.encode(enriched_text)),
            metadata={
                **doc.metadata,
                "raw_chunk": chunk_text,
                "context_sentence": context,
            },
        ))

    return chunks

