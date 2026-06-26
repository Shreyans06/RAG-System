import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.models import Chunk

def rerank(
        query: str,
        chunks: list[Chunk],
        api_key: str,
        top_n: int = 5,
        model: str = "rerank-v3.5",
) -> list[Chunk]:
    """
    Rerank a list of chunks based on their relevance to a query using Cohere's reranking API.
    Returns the top N most relevant chunks.
    """
    if not chunks:
        return []
    
    client = cohere.Client(api_key = api_key)
    results = _call_rerank(client, model, query, [chunk.content for chunk in chunks], top_n)
    return [chunks[result.index] for result in results.results]

@retry(stop = stop_after_attempt(3) , wait = wait_exponential(multiplier = 1 , min = 2 , max = 10))
def _call_rerank(client: cohere.Client, model: str , query: str , documents: list[str] , top_n: int):
    """
    Call Cohere's reranking API with retry logic.
    """
    return client.rerank(
        model = model,
        query = query,
        documents = documents,
        top_n = top_n,
    )

