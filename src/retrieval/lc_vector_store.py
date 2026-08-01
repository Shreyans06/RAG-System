from collections.abc import Callable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SparseVector

from src.ingestion.models import Chunk
from src.models.factory import get_embeddings, get_sparse_embeddings
from src.retrieval.vector_store import chunk_from_payload, hybrid_search


def _chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.content,
        metadata = {
            "chunk_id" : chunk.id,
            "document_id" : chunk.document_id,
            "parent_chunk_id" : chunk.parent_chunk_id,
            "content_type" : chunk.content_type.value,
            **chunk.metadata,
        },
    )

class QdrantHybridRetriever(BaseRetriever):
    """ LangChain BaseRetriever backed by Qdrant's native dense + sparse hybrid search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: QdrantClient
    collection_name: str
    top_k: int = 10
    filters: Filter | None = None
    dense_query_transform: Callable[[str], str] | None = None


    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        dense_model = get_embeddings()
        sparse_model = get_sparse_embeddings()

        dense_query_text = query
        if self.dense_query_transform is not None:
            try:
                dense_query_text = self.dense_query_transform(query)
            except Exception:
                dense_query_text = query  # fail open — e.g. HyDE call failed

        dense_vector = dense_model.embed_query(dense_query_text)
        sparse_raw = sparse_model.embed_query(query)
        sparse_vector = SparseVector(indices=sparse_raw.indices ,values=sparse_raw.values)

        points = hybrid_search(
            self.client,
            self.collection_name,
            dense_query_vector=dense_vector,
            sparse_query_vector=sparse_vector,
            top_k=self.top_k,
            filters=self.filters,
        )
        chunks = [chunk_from_payload(p.payload) for p in points]
        return [_chunk_to_document(c) for c in chunks]
