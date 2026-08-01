from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    session_id: str | None = None
    ticker: str | None = None
    filing_type: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return v
class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    processing_time_seconds: float
    errors: list[str] = []

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    citation_coverage: float
