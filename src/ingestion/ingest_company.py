import threading
from datetime import datetime, timezone

from qdrant_client.models import SparseVector

from src.config import get_settings
from src.ingestion.job_store import create_job, update_job
from src.ingestion.pipeline import ChunkingStrategy, IngestionPipeline
from src.ingestion.sec_edgar_client import SECEdgarClient
from src.ingestion.sec_metadata import SECMetadata
from src.models.factory import get_embeddings, get_sparse_embeddings
from src.retrieval.vector_store import create_collection_if_not_exists, make_client, upsert_chunks


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_sec_metadata(filing: dict) -> SECMetadata:
    return SECMetadata(
        company=filing["company"],
        ticker=filing["ticker"],
        cik=filing["cik"],
        filing_type=filing["filing_type"],
        filing_date=filing["filing_date"],
        fiscal_year=filing["fiscal_year"],
        fiscal_period=filing["fiscal_period"],
        accession_number=filing["accession_number"],
        source_url=(
            f"https://www.sec.gov/Archives/edgar/data/{int(filing['cik'])}/"
            f"{filing['accession_number'].replace('-', '')}/{filing['primary_document']}"
        ),
    )


def ingest_company_filings(ticker: str, filing_types: list[str], years: list[int]) -> dict:
    """Fetch and ingest a company's SEC filings into Qdrant. Returns a summary dict.
    Runs synchronously and blocks for the duration — callers that need this to not
    block the caller should use start_ingestion_job() instead."""
    settings = get_settings()
    edgar = SECEdgarClient(user_agent=settings.SEC_EDGAR_USER_AGENT, cache_dir="data/raw")
    filings = edgar.get_filings(ticker, filing_types=filing_types, years=years)

    if not filings:
        return {"filings_found": 0, "chunks_upserted": 0, "errors": []}

    qdrant = make_client(settings.QDRANT_URL)
    create_collection_if_not_exists(
        qdrant, settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.retrieval.get("embedding_dimensions", 3072),
    )
    pipeline = IngestionPipeline(
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        strategy=ChunkingStrategy.HIERARCHICAL,
        parent_chunk_size=settings.ingestion.get("parent_chunk_size", 1000),
        child_chunk_size=settings.ingestion.get("child_chunk_size", 200),
        chunk_overlap=settings.ingestion.get("chunk_overlap", 20),
        contextual_summary_model=settings.ingestion.get("contextual_summary_model", "claude-haiku-4-5"),
    )

    total_chunks = 0
    errors: list[str] = []
    for filing in filings:
        try:
            path = edgar.download_filing(filing)
            sec_meta = _build_sec_metadata(filing)
            result = pipeline.ingest(path, sec_metadata=sec_meta)
            chunks = result.chunks
            errors.extend(result.errors)
            if not chunks:
                continue
            texts = [c.content for c in chunks]
            dense_embeddings = get_embeddings().embed_documents(texts)
            sparse_raw = get_sparse_embeddings().embed_documents(texts)
            sparse_embeddings = [SparseVector(indices=v.indices, values=v.values) for v in sparse_raw]
            upsert_chunks(
                qdrant, settings.QDRANT_COLLECTION_NAME, chunks,
                dense_embeddings, sparse_embeddings=sparse_embeddings,
            )
            total_chunks += len(chunks)
        except Exception as e:
            errors.append(f"{filing['ticker']} {filing['filing_type']} {filing['accession_number']}: {e}")

    return {"filings_found": len(filings), "chunks_upserted": total_chunks, "errors": errors}


def _run_job(job_id: str, ticker: str, filing_types: list[str], years: list[int]) -> None:
    try:
        result = ingest_company_filings(ticker, filing_types, years)
        update_job(job_id, status="done", result=result, finished_at=_now())
    except Exception as e:
        update_job(job_id, status="failed", error=str(e), finished_at=_now())


def start_ingestion_job(ticker: str, filing_types: list[str], years: list[int]) -> str:
    """Kick off ingestion in a background thread, returning immediately with a job_id."""
    job_id = create_job(ticker, filing_types, years)
    thread = threading.Thread(target=_run_job, args=(job_id, ticker, filing_types, years), daemon=True)
    thread.start()
    return job_id
