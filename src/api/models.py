from enum import Enum

from pydantic import BaseModel


class RetrievalMode(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    processing_time_seconds: float


class QueryRequest(BaseModel):
    question: str
    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    citation_coverage: float
