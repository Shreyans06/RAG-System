import argparse
import sys
import time

from qdrant_client.models import SparseVector

from src.config import get_settings
from src.ingestion.pipeline import ChunkingStrategy, IngestionPipeline
from src.ingestion.sec_edgar_client import SECEdgarClient
from src.ingestion.sec_metadata import SECMetadata
from src.models.factory import get_embeddings, get_sparse_embeddings
from src.retrieval.vector_store import create_collection_if_not_exists, make_client, upsert_chunks


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

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-ingest SEC filings for a ticker into Qdrant.")
    parser.add_argument("--ticker", required=True, help="e.g. AAPL")
    parser.add_argument("--years", required=True, help="comma-separated fiscal years, e.g. 2023,2024")
    parser.add_argument("--filing-types", default="10-K,10-Q,8-K", help="comma-separated, default 10-K,10-Q,8-K")
    parser.add_argument(
        "--strategy", default="hierarchical", choices=["hierarchical", "contextual"],
        help="hierarchical is free (no LLM calls); contextual makes one paid Claude call per "
             "chunk (Anthropic's contextual retrieval pattern). Default: hierarchical.",
    )
    parser.add_argument("--dry-run", action="store_true", help="list matching filings without downloading/ingesting")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    filing_types = args.filing_types.split(",")
    strategy = ChunkingStrategy.CONTEXTUAL if args.strategy == "contextual" else ChunkingStrategy.HIERARCHICAL

    settings = get_settings()
    edgar = SECEdgarClient(user_agent=settings.SEC_EDGAR_USER_AGENT, cache_dir="data/raw")

    filings = edgar.get_filings(args.ticker, filing_types=filing_types, years=years)
    print(f"Found {len(filings)} filing(s) for {args.ticker.upper()} matching years={years}, types={filing_types}\n")
    for f in filings:
        print(
            f"  {f['filing_type']:6s} FY{f['fiscal_year']} {f['fiscal_period']:3s}  "
            f"filed {f['filing_date']}  ({f['accession_number']})"
        )

    if args.dry_run:
        print("\n--dry-run: nothing downloaded or ingested.")
        return 0

    if not filings:
        print("Nothing to ingest.")
        return 0

    qdrant = make_client(settings.QDRANT_URL)
    create_collection_if_not_exists(
        qdrant, settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.retrieval.get("embedding_dimensions", 3072),
    )

    pipeline = IngestionPipeline(
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        strategy=strategy,
        parent_chunk_size=settings.ingestion.get("parent_chunk_size", 1000),
        child_chunk_size=settings.ingestion.get("child_chunk_size", 200),
        chunk_overlap=settings.ingestion.get("chunk_overlap", 20),
        contextual_summary_model=settings.ingestion.get("contextual_summary_model", "claude-haiku-4-5"),
    )

    total_chunks = 0
    errors: list[tuple[dict, str]] = []
    print()
    for i, filing in enumerate(filings, start=1):
        label = f"{filing['ticker']} {filing['filing_type']} FY{filing['fiscal_year']} {filing['fiscal_period']}"
        print(f"[{i}/{len(filings)}] {label} ...", end=" ", flush=True)
        start = time.perf_counter()
        try:
            path = edgar.download_filing(filing)
            sec_meta = _build_sec_metadata(filing)
            result = pipeline.ingest(path, sec_metadata=sec_meta)
            chunks = result.chunks
            if result.errors:
                print(f"  [partial errors: {'; '.join(result.errors)}]")

            if not chunks:
                print("0 chunks, skipping upsert")
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
            print(f"{len(chunks)} chunks in {time.perf_counter() - start:.1f}s")
        except Exception as e:  # noqa: BLE001 - one bad filing must not abort the whole batch
            print(f"FAILED: {e}")
            errors.append((filing, str(e)))

    print(f"\nDone. {total_chunks} total chunks upserted across {len(filings) - len(errors)} filing(s).")
    if errors:
        print(f"\n{len(errors)} filing(s) failed:")
        for filing, err in errors:
            print(f"  {filing['ticker']} {filing['filing_type']} {filing['accession_number']}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

