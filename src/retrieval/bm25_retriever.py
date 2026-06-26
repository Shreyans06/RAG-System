import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.models import Chunk

class BM25Retriever:
    def __init__(self):
        self.corpus_chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
    
    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Build the BM25 index from a list of chunks.
        """
        self.corpus_chunks = chunks
        tokenized = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk,float]]:
        """
        Search for the top K most relevant chunks based on a query string.
        Returns a list of tuples containing the chunk and it's BM25 score.
        """
        if self.bm25 is None or not self.corpus_chunks:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.corpus_chunks[i] , float(scores[i])) for i in top_indices]
    
    def save(self, path: str | Path) -> None:
        """
        Save the BM25 index and corpus chunks to a file.
        """
        with open(path , "wb" ) as f:
            pickle.dump(self , f)
    
    @staticmethod
    def load(path: str | Path) -> "BM25Retriever":
        """
        Load the BM25 index and corpus chunks from a file.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
        
    


    
