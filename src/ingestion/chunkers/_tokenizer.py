from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def split_into_token_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks
