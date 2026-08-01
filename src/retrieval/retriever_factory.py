from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from src.config import get_settings
from src.models.factory import get_chat_model, get_reranker
from src.retrieval.hyde import generate_hypothetical_answer
from src.retrieval.lc_vector_store import QdrantHybridRetriever


def build_retriever(client: QdrantClient, collection_name: str, filters: Filter | None = None):
    """Composes the full retrieval pipeline: base hybrid retriever -> optional HyDE (dense
    leg only) -> optional multi-query expansion -> reranking."""

    settings = get_settings()

    top_k = settings.retrieval.get("dense_top_k", 20)
    retriever = QdrantHybridRetriever(
        client=client, collection_name=collection_name, top_k=top_k, filters=filters
    )

    if settings.retrieval.get("use_hyde", False):
        retriever.dense_query_transform = generate_hypothetical_answer

    if settings.retrieval.get("use_multi_query", False):
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=get_chat_model("query_transform"))

    from langchain_classic.retrievers import ContextualCompressionRetriever
    reranker = get_reranker()
    if hasattr(reranker, "top_n"):
        reranker.top_n = settings.retrieval.get("rerank_top_n", 5)

    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)

