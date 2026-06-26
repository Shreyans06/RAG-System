import time 
from tenacity import retry, stop_after_attempt, wait_exponential
import openai

def embed_texts(
        texts: list[str],
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI's embedding API with retry logic.
    """
    client = openai.OpenAI(api_key = api_key)
    all_embeddings: list[list[float]] = []

    clean_texts = [t if t.strip() else " " for t in texts]

    for i in range(0 , len(clean_texts), batch_size):
        batch = clean_texts[i : i + batch_size]
        response = _create_embeddings(client , model , batch)
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)
        if i + batch_size < len(clean_texts):
            time.sleep(0.2)
    
    return all_embeddings

def embed_query(text: str, api_key: str, model: str = "text-embedding-3-large") -> list[float]:
    """
    Embed a single query text using OpenAI's embedding API
    """
    return embed_texts([text] , api_key = api_key , model = model)[0]

@retry(stop = stop_after_attempt(3) , wait = wait_exponential(multiplier = 1 , min = 2 , max = 10))
def _create_embeddings(client: openai.OpenAI, model: str, batch: list[str]):
    """
    Create embeddings for a batch of texts using OpenAI's embedding API with retry logic.
    """
    response = client.embeddings.create(model = model , input = batch)
    return response