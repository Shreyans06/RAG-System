from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from src.ingestion.models import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embeddings import embed_query
from src.retrieval.vector_store import search as vector_search

def reciprocal_rank_fusion(
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int = 60,
) -> list[str]:
    """
    Perform Reciprocal Rank Fusion (RRF) on dense and sparse search results.
    Returns a list of document IDs ranked by their RRF score.
    """
    fused: dict[str , float] = {}

    for rank, (chunk_id , _) in enumerate(dense_results):
        fused[chunk_id] = fused.get(chunk_id , 0.0) + 1.0 / (k + rank + 1)
    
    for rank, (chunk_id , _) in enumerate(sparse_results):
        fused[chunk_id] = fused.get(chunk_id , 0.0) + 1.0 / (k + rank + 1)
    
    return sorted(fused , key=lambda x: fused[x] , reverse = True)

class HybridRetriever:
    def __init__(
            self,
            qdrant_client: QdrantClient,
            collection_name: str,
            bm25_retriever: BM25Retriever,
            openai_api_key: str,
            embedding_model: str = "text-embedding-3-large",
            top_k: int = 20,
            filters: Filter | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.bm25_retriever = bm25_retriever
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.filters = filters

        # chunk_id -> Chunk lookup built from BM25 corpus
        self._lookup: dict[str, Chunk] = {
            chunk.id: chunk for chunk in bm25_retriever.corpus_chunks
        }
    
    def retrieve(self, query: str) -> list[Chunk]:
        """
        Retrieve relevant chunks for a given query using both dense and sparse retrieval methods.
        Returns a list of Chunks ranked by their relevance.
        """
        # Dense retrieval
        query_vector = embed_query(query , api_key= self.openai_api_key , model = self.embedding_model)
        dense_hits = vector_search(self.qdrant_client , self.collection_name , query_vector , self.top_k , self.filters)
        dense_results = [(hit.payload["chunk_id"], hit.score) for hit in dense_hits if hit is not None and hit.payload is not None]

        # Sparse retrieval
        sparse_hits = self.bm25_retriever.search(query , self.top_k)
        sparse_results = [(chunk.id , score) for chunk , score in sparse_hits]

        # Fuse and reconstruct
        fused_ids = reciprocal_rank_fusion(dense_results, sparse_results)[: self.top_k]
        return [self._lookup[chunk_id] for chunk_id in fused_ids if chunk_id in self._lookup]
    

