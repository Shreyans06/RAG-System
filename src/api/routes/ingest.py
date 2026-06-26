import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.api.models import IngestResponse
from src.ingestion.pipeline import ChunkingStrategy, IngestionPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embeddings import embed_texts
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import (
    create_collection_if_not_exists,
    delete_collection,
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
        chunks = pipeline.ingest(tmp_path)

    embeddings = embed_texts(
        [c.content for c in chunks],
        api_key=settings.OPENAI_API_KEY,
        model=settings.retrieval.get("embedding_model", "text-embedding-3-large"),
    )

    upsert_chunks(request.app.state.qdrant, settings.QDRANT_COLLECTION_NAME, chunks, embeddings)

    request.app.state.all_chunks.extend(chunks)
    request.app.state.ingested_files.add(file.filename)
    request.app.state.bm25.build_index(request.app.state.all_chunks)

    request.app.state.hybrid_retriever = HybridRetriever(
        qdrant_client=request.app.state.qdrant,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        bm25_retriever=request.app.state.bm25,
        openai_api_key=settings.OPENAI_API_KEY,
        embedding_model=settings.retrieval.get("embedding_model", "text-embedding-3-large"),
        top_k=settings.retrieval.get("dense_top_k", 20),
    )

    return IngestResponse(
        filename=file.filename,
        chunks_created=len(chunks),
        processing_time_seconds=round(time.perf_counter() - start, 2),
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
    request.app.state.all_chunks = []
    request.app.state.ingested_files = set()
    request.app.state.bm25 = BM25Retriever()
    if hasattr(request.app.state, "hybrid_retriever"):
        del request.app.state.hybrid_retriever
    return {"cleared": True}


@router.get("/documents")
async def list_documents(request: Request) -> dict:
    return {"files": sorted(request.app.state.ingested_files)}


@router.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request) -> dict:
    settings = request.app.state.settings
    if filename not in request.app.state.ingested_files:
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    # Remove from Qdrant by filtering on payload filename
    request.app.state.qdrant.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
    )

    # Rebuild BM25 without deleted file's chunks
    request.app.state.all_chunks = [
        c for c in request.app.state.all_chunks if c.metadata.get("filename") != filename
    ]
    request.app.state.ingested_files.discard(filename)

    if request.app.state.all_chunks:
        request.app.state.bm25.build_index(request.app.state.all_chunks)
        request.app.state.hybrid_retriever = HybridRetriever(
            qdrant_client=request.app.state.qdrant,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            bm25_retriever=request.app.state.bm25,
            openai_api_key=settings.OPENAI_API_KEY,
            embedding_model=settings.retrieval.get("embedding_model", "text-embedding-3-large"),
            top_k=settings.retrieval.get("dense_top_k", 20),
        )
    else:
        request.app.state.bm25 = BM25Retriever()
        if hasattr(request.app.state, "hybrid_retriever"):
            del request.app.state.hybrid_retriever

    return {"deleted": filename}
