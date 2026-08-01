from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.config import get_settings
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
