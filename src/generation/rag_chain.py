from collections.abc import Generator
from typing import Any

from src.api.models import RetrievalMode
from src.generation.llm import LLMClient
from src.generation.prompt_templates import build_messages
from src.ingestion.models import Chunk
from src.retrieval import reranker
from src.retrieval.embeddings import embed_query
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import chunk_from_payload, search as vector_search


def _citation_coverage(response: str, chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    cited = sum(1 for chunk in chunks if chunk.id[:8] in response)
    return cited / len(chunks)


class RAGChain:
    def __init__(
        self,
        llm: LLMClient,
        hybrid_retriever: HybridRetriever,
        cohere_api_key: str,
        rerank_top_n: int = 5,
        rerank_model: str = "rerank-v3.5",
    ) -> None:
        self.llm = llm
        self.hybrid_retriever = hybrid_retriever
        self.cohere_api_key = cohere_api_key
        self.rerank_top_n = rerank_top_n
        self.rerank_model = rerank_model

    def _retrieve(self, question: str, mode: RetrievalMode) -> tuple[list[Chunk], list[dict]]:
        if mode == RetrievalMode.DENSE:
            query_vector = embed_query(
                question,
                api_key=self.hybrid_retriever.openai_api_key,
                model=self.hybrid_retriever.embedding_model,
            )
            hits = vector_search(
                self.hybrid_retriever.qdrant_client,
                self.hybrid_retriever.collection_name,
                query_vector,
                self.rerank_top_n,
            )
            chunks = [chunk_from_payload(h.payload) for h in hits if h.payload]

        elif mode == RetrievalMode.HYBRID:
            candidates = {c.id: c for c in self.hybrid_retriever.retrieve(question)}
            chunks = list(candidates.values())[: self.rerank_top_n]

        else:  # HYBRID_RERANK
            candidates = {c.id: c for c in self.hybrid_retriever.retrieve(question)}
            chunks = reranker.rerank(
                query=question,
                chunks=list(candidates.values()),
                api_key=self.cohere_api_key,
                top_n=self.rerank_top_n,
                model=self.rerank_model,
            )

        sources = [{"chunk_id": c.id[:8], **c.metadata} for c in chunks]
        return chunks, sources

    def query(self, question: str, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK) -> dict[str, Any]:
        chunks, sources = self._retrieve(question, mode)
        messages, system = build_messages(question, chunks)
        response = self.llm.generate(system, messages)
        return {
            "answer": response,
            "sources": sources,
            "citation_coverage": _citation_coverage(response, chunks),
            "mode": mode,
        }

    def stream(self, question: str, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK) -> Generator[str | dict, None, None]:
        chunks, sources = self._retrieve(question, mode)
        messages, system = build_messages(question, chunks)
        yield from self.llm.stream(system, messages)
        yield {"sources": sources, "mode": mode}
