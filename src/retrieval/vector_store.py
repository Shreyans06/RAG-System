import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    Fusion,
    FusionQuery,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.ingestion.models import Chunk

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_UPSERT_BATCH_SIZE = 100


_KEYWORD_INDEX_FIELDS = ["ticker", "filing_type", "fiscal_period", "cik"]
_INTEGER_INDEX_FIELDS = ["fiscal_year"]

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
            vectors_config = {
                DENSE_VECTOR_NAME: VectorParams(size = vector_size , distance= Distance.COSINE),
            },
            sparse_vectors_config= {
                SPARSE_VECTOR_NAME : SparseVectorParams(modifier= Modifier.IDF),
            },
        )
    _ensure_payload_indexes(client , collection_name)

def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    for field in _KEYWORD_INDEX_FIELDS:
        client.create_payload_index(collection_name, field_name=field, field_schema=PayloadSchemaType.KEYWORD)
    for field in _INTEGER_INDEX_FIELDS:
        client.create_payload_index(collection_name, field_name=field, field_schema=PayloadSchemaType.INTEGER)

def upsert_chunks(
        client: QdrantClient,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector] | None = None
) -> None:
    """
    Upserts chunks with a dense vector and, if provided, a sparse vector on the same point into a Qdrant collection.
    """

    points = []
    for i , (chunk , dense_vec) in enumerate(zip(chunks , embeddings)):
        vector: dict[str , object] = {DENSE_VECTOR_NAME: dense_vec}
        if sparse_embeddings is not None:
            vector[SPARSE_VECTOR_NAME] = sparse_embeddings[i]
        points.append(
            PointStruct(
                id = _to_uuid(chunk.id),
                vector = vector,
                payload={
                    "content": chunk.content,
                    "chunk_id" : chunk.id,
                    "document_id" : chunk.document_id,
                    "parent_chunk_id" : chunk.parent_chunk_id,
                    "content_type" : chunk.content_type,
                    "token_count" : chunk.token_count,
                    **chunk.metadata,
                }
            )
        )
    # Qdrant's REST API caps request payloads at 32MB; a single large filing's worth of
    # points (dense + sparse vectors + full chunk text) can exceed that in one call, so
    # upsert in batches rather than all-at-once.
    for i in range(0, len(points), _UPSERT_BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=points[i:i + _UPSERT_BATCH_SIZE])

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
        using = DENSE_VECTOR_NAME,
        limit=top_k,
        query_filter=filters,
        with_payload=True,
    )
    return result.points

def hybrid_search(
        client: QdrantClient,
        collection_name: str,
        dense_query_vector: list[float],
        sparse_query_vector: SparseVector,
        top_k: int = 5,
        filters: Filter | None = None,
) -> list[ScoredPoint]:
    """ Hybrid dense + sparse search."""
    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense_query_vector , using=DENSE_VECTOR_NAME , filter= filters, limit=top_k * 4),
            Prefetch(query=sparse_query_vector, using=SPARSE_VECTOR_NAME, filter= filters, limit=top_k * 4),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
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

def load_all_chunks(client: QdrantClient, collection_name: str, batch_size: int = 256) -> list[Chunk]:
    """
    Load all chunks from a Qdrant collection in batches.
    """
    chunks: list[Chunk] = []
    offset = None
    while True:
        points , offset = client.scroll(
            collection_name = collection_name,
            limit = batch_size,
            offset = offset,
            with_payload = True,
            with_vectors = False,
        )
        chunks.extend(chunk_from_payload(p.payload) for p in points)
        if offset is None:
            break
    
    return chunks

def list_distinct_payload_values(client: QdrantClient, collection_name: str, field: str, batch_size: int = 256) -> list:
    """Scans the collection for distinct values of a payload field — replaces maintaining a
    separate in-memory set of ingested filenames/tickers."""
    values = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name, limit=batch_size, offset=offset,
            with_payload=[field], with_vectors=False,
        )
        values.update(p.payload[field] for p in points if p.payload.get(field) is not None)
        if offset is None:
            break
    return sorted(values)
