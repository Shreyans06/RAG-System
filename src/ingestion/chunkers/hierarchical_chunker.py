from src.ingestion.chunkers._tokenizer import get_tokenizer, split_into_token_windows
from src.ingestion.models import Chunk, IngestionItem


def hierarchical_chunker(
    doc: IngestionItem,
    parent_size: int = 1000,
    child_size: int = 200,
    overlap: int = 20,
) -> list[Chunk]:
    """
    Hierarchically chunk a document into parent and child chunks based on token windows.
    """
    chunks = []
    tokenizer = get_tokenizer()

    for p_idx, parent_text in enumerate(split_into_token_windows(doc.content, parent_size, overlap)):
        parent_chunk_id = f"{doc.id}-p{p_idx}"

        for c_idx, child_text in enumerate(split_into_token_windows(parent_text, child_size, overlap)):
            chunks.append(Chunk(
                id = f"{parent_chunk_id}-c{c_idx}",
                document_id = doc.id,
                parent_chunk_id = parent_chunk_id,
                content = child_text,
                content_type = doc.content_type,
                token_count = len(tokenizer.encode(child_text)),
                metadata = {
                    **doc.metadata,
                    "parent_text" : parent_text,
                    "parent_chunk_id" : parent_chunk_id,
                },
            ))

    return chunks
