import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, ScoredPoint, VectorParams

from src.ingestion.models import Chunk

def _to_uuid(chunk_id: str) -> str:
    """
    Convert a chunk ID to a UUID string.
    """
    try:
        return str(uuid.UUID(chunk_id))
    except ValueError:
        # If the chunk_id is not a valid UUID, generate a new UUID based on the chunk_id
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

def make_client(url: str, api_key: str = "") -> QdrantClient:
    """
    Create a Qdrant client with the given URL and API key.
    """
    return QdrantClient(url=url, api_key=api_key or None)

def create_collection_if_not_exists(
        client: QdrantClient,
        collection_name: str,
        vector_size: int = 3072,
) -> None:
    """
    Create a Qdrant collection if it does not alreay exist.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(size = vector_size , distance = Distance.COSINE),
        )

def upsert_chunks(
        client: QdrantClient,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
) -> None:
    """
    Upsert a list of chunks and their corresponding embeddings into a Qdrant collection.
    """
    points = [
        PointStruct(
            id = _to_uuid(chunk.id),
            vector = vector,
            payload = {
                "content" : chunk.content,
                "chunk_id" : chunk.id,
                "document_id" : chunk.document_id,
                "parent_chunk_id" : chunk.parent_chunk_id,
                "content_type" : chunk.content_type,
                "token_count" : chunk.token_count,
                **chunk.metadata,
            },
        )
        for chunk , vector in zip(chunks, embeddings)
    ]
    client.upsert(collection_name = collection_name, points = points)

def search(
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: Filter | None = None,
) -> list[ScoredPoint]:
    """
    Search for the top K most similar chunks in a Qdrant collection based on a query vector.
    """
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=filters,
        with_payload=True,
    )
    return result.points

def chunk_from_payload(payload: dict) -> Chunk:
    from src.ingestion.models import ContentType
    reserved = {"chunk_id", "document_id", "parent_chunk_id", "content", "content_type", "token_count"}
    return Chunk(
        id=payload["chunk_id"],
        document_id=payload["document_id"],
        parent_chunk_id=payload.get("parent_chunk_id"),
        content=payload["content"],
        content_type=ContentType(payload["content_type"]),
        token_count=payload.get("token_count", 0),
        metadata={k: v for k, v in payload.items() if k not in reserved},
    )


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Delete a Qdrant collection.
    """
    client.delete_collection(collection_name = collection_name)