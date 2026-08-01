import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector

from src.api.models import IngestResponse
from src.ingestion.pipeline import ChunkingStrategy, IngestionPipeline
from src.models.factory import get_embeddings, get_sparse_embeddings
from src.retrieval.vector_store import (
    create_collection_if_not_exists,
    delete_collection,
    list_distinct_payload_values,
    upsert_chunks,
)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile, request: Request) -> IngestResponse:
    settings = request.app.state.settings
    max_bytes = settings.ingestion.get("max_upload_size_mb", 50) * 1024 * 1024

    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File exceeds maximum upload size")

    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(contents)

        pipeline = IngestionPipeline(
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            strategy=ChunkingStrategy.HIERARCHICAL,
            parent_chunk_size=settings.ingestion.get("parent_chunk_size", 1000),
            child_chunk_size=settings.ingestion.get("child_chunk_size", 200),
            chunk_overlap=settings.ingestion.get("chunk_overlap", 20),
        )
        result = pipeline.ingest(tmp_path)

    chunks = result.chunks
    if not chunks:
        detail = "; ".join(result.errors) or "no extractable content"
        raise HTTPException(status_code=422, detail=f"Ingestion produced no chunks: {detail}")

    dense_model = get_embeddings()
    dense_embeddings = dense_model.embed_documents([c.content for c in chunks])

    sparse_model = get_sparse_embeddings()
    sparse_raw = sparse_model.embed_documents([c.content for c in chunks])
    sparse_embeddings = [SparseVector(indices=v.indices, values=v.values) for v in sparse_raw]

    upsert_chunks(
        request.app.state.qdrant, settings.QDRANT_COLLECTION_NAME, chunks,
        dense_embeddings, sparse_embeddings=sparse_embeddings,
    )

    return IngestResponse(
        filename=file.filename,
        chunks_created=len(chunks),
        processing_time_seconds=round(time.perf_counter() - start, 2),
        errors=result.errors,
    )



@router.get("/status")
async def status(request: Request) -> dict:
    settings = request.app.state.settings
    count = request.app.state.qdrant.count(settings.QDRANT_COLLECTION_NAME).count
    return {"vector_count": count, "has_documents": count > 0}


@router.delete("/documents")
async def clear_documents(request: Request) -> dict:
    settings = request.app.state.settings
    delete_collection(request.app.state.qdrant, settings.QDRANT_COLLECTION_NAME)
    create_collection_if_not_exists(
        request.app.state.qdrant,
        settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.retrieval.get("embedding_dimensions", 3072),
    )
    return {"cleared": True}


@router.get("/documents")
async def list_documents(request: Request) -> dict:
    settings = request.app.state.settings
    files = list_distinct_payload_values(request.app.state.qdrant, settings.QDRANT_COLLECTION_NAME, "filename")
    return {"files": files}

@router.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request) -> dict:
    settings = request.app.state.settings
    files = list_distinct_payload_values(request.app.state.qdrant, settings.QDRANT_COLLECTION_NAME, "filename")
    if filename not in files:
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    request.app.state.qdrant.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))]),
    )
    return {"deleted": filename}
