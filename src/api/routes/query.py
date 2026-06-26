import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models import QueryRequest, RetrievalMode
from src.generation.rag_chain import RAGChain

router = APIRouter()


def _make_chain(request: Request) -> RAGChain:
    return RAGChain(
        llm=request.app.state.llm,
        hybrid_retriever=request.app.state.hybrid_retriever,
        cohere_api_key=request.app.state.settings.COHERE_API_KEY,
        rerank_top_n=request.app.state.settings.retrieval.get("rerank_top_n", 5),
    )


def _guard(request: Request) -> None:
    if not hasattr(request.app.state, "hybrid_retriever"):
        raise HTTPException(status_code=400, detail="No documents ingested yet")


@router.post("/query")
async def query(request_body: QueryRequest, request: Request) -> StreamingResponse:
    _guard(request)
    chain = _make_chain(request)

    def event_stream():
        for chunk in chain.stream(request_body.question, mode=request_body.mode):
            if isinstance(chunk, str):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            elif isinstance(chunk, dict):
                yield f"data: {json.dumps({'done': True, **chunk})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/query/compare")
async def compare(request_body: QueryRequest, request: Request) -> dict:
    _guard(request)
    chain = _make_chain(request)
    return {
        mode.value: chain.query(request_body.question, mode=mode)
        for mode in RetrievalMode
    }
