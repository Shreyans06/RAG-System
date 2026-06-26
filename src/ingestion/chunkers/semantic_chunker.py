import anthropic

from src.ingestion.chunkers._tokenizer import get_tokenizer, split_into_token_windows
from src.ingestion.models import Chunk , IngestionItem

_CONTEXT_PROMPT = """\
Here is a document excerpt. Write ONE short sentence (max 20 words) that
situates this excerpt in the overall document for retrieval purposes.
Do NOT summarize - give context only.

<excerpt>{chunk_text}</excerpt>

Context sentence:
"""

def _call_claude(client: anthropic.Anthropic, model: str, chunk_text: str) -> str:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=60,
            messages=[{"role" : "user" , "content" : _CONTEXT_PROMPT.format(chunk_text = chunk_text[:800])}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"Error calling Claude API: {e}")

def chunk_with_context(
    doc: IngestionItem,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    chunk_size: int = 400,
    overlap: int = 40,
) -> list[Chunk]:
    """
    Chunk a document into smaller chunks and generate context sentences for each chunk.
    """
    chunks = []
    tokenizer = get_tokenizer()
    client = anthropic.Anthropic(api_key=api_key)

    for idx, chunk_text in enumerate(split_into_token_windows(doc.content, chunk_size, overlap)):
        context = _call_claude(client, model, chunk_text)
        enriched_text = f"{context}\n\n{chunk_text}" if context else chunk_text

        chunks.append(Chunk(
            id = f"{doc.id}-ctx{idx}",
            document_id = doc.id,
            parent_chunk_id = None,
            content = enriched_text,
            content_type = doc.content_type,
            token_count = len(tokenizer.encode(enriched_text)),
            metadata = {
                **doc.metadata,
                "raw_chunk" : chunk_text,
                "context_sentence" : context,
            },
        ))
    
    return chunks

