from langchain_core.tools import tool

from src.ingestion.ingest_company import start_ingestion_job
from src.ingestion.job_store import get_latest_job_for_ticker


@tool
def ingest_company_filings_tool(ticker: str, filing_types: list[str], years: list[int]) -> str:
    """Start ingesting a company's SEC filings (10-K/10-Q) into the knowledge base, so
    questions about that company can be answered. Use this when the user asks to add,
    ingest, or include a company's filings — not when they're asking a question about an
    already-ingested company. This runs in the background and takes several minutes; it
    does not block, so tell the user it has started and they can ask for a status update
    later.

    Args:
        ticker: Stock ticker symbol, e.g. "TSLA" for Tesla.
        filing_types: Filing types to ingest, e.g. ["10-K", "10-Q"].
        years: Fiscal years to ingest, e.g. [2023, 2024, 2025].
    """
    job_id = start_ingestion_job(ticker.upper(), filing_types, years)
    return f"Started ingesting {ticker.upper()} ({', '.join(filing_types)}, years {years}). Job ID: {job_id}."


@tool
def check_ingestion_status_tool(ticker: str) -> str:
    """Check the status of the most recent ingestion job for a company. Use this when
    the user asks whether a previously-requested ingestion is done, e.g. "is Tesla ready
    yet?" or "did the Microsoft ingestion finish?".

    Args:
        ticker: Stock ticker symbol, e.g. "TSLA".
    """
    job = get_latest_job_for_ticker(ticker.upper())
    if job is None:
        return f"No ingestion job found for {ticker.upper()}."
    if job["status"] == "running":
        return f"Ingestion for {ticker.upper()} is still in progress (started {job['started_at']})."
    if job["status"] == "failed":
        return f"Ingestion for {ticker.upper()} failed: {job['error']}"
    result = job["result"]
    return (
        f"Ingestion for {ticker.upper()} finished: {result['chunks_upserted']} chunks "
        f"upserted across {result['filings_found'] - len(result['errors'])} filing(s)."
    )
