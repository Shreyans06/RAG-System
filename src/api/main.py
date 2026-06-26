from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.config import get_settings
from src.generation.llm import LLMClient
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import create_collection_if_not_exists, make_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    qdrant = make_client(settings.QDRANT_URL)
    create_collection_if_not_exists(
        qdrant,
        settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.retrieval.get("embedding_dimensions", 3072),
    )

    app.state.settings = settings
    app.state.qdrant = qdrant
    app.state.bm25 = BM25Retriever()
    app.state.all_chunks = []
    app.state.ingested_files: set[str] = set()
    app.state.llm = LLMClient(
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.generation.get("model", "claude-sonnet-4-6"),
        max_tokens=settings.generation.get("max_tokens", 2048),
        temperature=settings.generation.get("temperature", 0.0),
    )

    yield


app = FastAPI(title="RAG System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api")
app.include_router(query_router, prefix="/api")
