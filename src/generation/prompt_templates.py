from anthropic.types import MessageParam

from src.ingestion.models import Chunk

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based strictly on the provided context.

Rules:
1. Only use information from the provided context to answer questions.
2. After EVERY factual claim, add a citation using the chunk ID shown in the context, e.g. [a1b2c3d4].
3. If the context contains a table or image description relevant to the answer, reference it.
4. If the answer is not in the context, say exactly "I don't have enough information to answer this."
5. Do not make up information or use prior knowledge.

Context:
{context}
"""

def build_context_string(chunks: list[Chunk]) -> str:
    """
    Build a context string from a list of chunks for use in prompt templates.
    """
    context_parts = []
    for chunk in chunks:
        text = chunk.metadata.get("parent_text", chunk.content)
        citation_id = chunk.id[:8]
        filename = chunk.metadata.get("filename" , "")
        page = chunk.metadata.get("page" , "")
        header = f"[{citation_id}] ({filename} , page {page})" if page else f"[{citation_id}] ({filename})"
        context_parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(context_parts)

def build_messages(
        query: str, chunks: list[Chunk]
) -> tuple[list[MessageParam], str]:
    """
    Build a list of messages for the LLM prompt based on the query and context chunks.
    Returns a tuple of (messages, context_string).
    """
    system = SYSTEM_PROMPT.format(context = build_context_string(chunks))
    messages: list[MessageParam] = [{"role": "user", "content": query}]
    return messages, system