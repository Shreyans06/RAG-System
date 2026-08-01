import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models import QueryRequest
from src.generation.lc_rag_chain import LCRAGChain
from src.retrieval.filters import build_filter

router = APIRouter()


def _guard(request: Request) -> None:
    settings = request.app.state.settings
    count = request.app.state.qdrant.count(settings.QDRANT_COLLECTION_NAME).count
    if count == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet")


def _explicit_filters(request_body: QueryRequest):
    if not any([request_body.ticker, request_body.filing_type, request_body.fiscal_year, request_body.fiscal_period]):
        return None
    tickers = [request_body.ticker] if request_body.ticker else None
    return build_filter(
        tickers=tickers,
        filing_type=request_body.filing_type,
        fiscal_year=request_body.fiscal_year,
        fiscal_period=request_body.fiscal_period,
    )


@router.post("/query")
async def query(request_body: QueryRequest, request: Request) -> StreamingResponse:
    _guard(request)
    settings = request.app.state.settings
    chain = LCRAGChain(request.app.state.qdrant, settings.QDRANT_COLLECTION_NAME)
    filters = _explicit_filters(request_body)

    def event_stream():
        for chunk in chain.stream(request_body.question, session_id= request_body.session_id, filters=filters):
            if isinstance(chunk, str):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            elif isinstance(chunk, dict):
                yield f"data: {json.dumps({'done': True, **chunk})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
