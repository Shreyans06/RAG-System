import sys

from src.config import get_settings
from src.ingestion.batch_embeddings import (
    complete_batch_job,
    list_pending_jobs,
    list_queued_labels,
    submit_queued_batches,
)
from src.retrieval.vector_store import make_client


def main() -> int:
    settings = get_settings()
    qdrant = make_client(settings.QDRANT_URL)

    jobs = list_pending_jobs()
    still_pending = 0
    for job in jobs:
        print(f"{job['label']} (batch {job['batch_id']}, {job['num_chunks']} chunks) ... ", end="", flush=True)
        status = complete_batch_job(job, qdrant, settings.QDRANT_COLLECTION_NAME)
        print(status)
        if not status.startswith("completed") and "re-queued" not in status:
            still_pending += 1

    submitted = submit_queued_batches()
    if submitted:
        print(f"\nSubmitted next queued ticker: {submitted}")
    else:
        remaining = list_queued_labels()
        if remaining:
            print(f"\n{len(remaining)} ticker(s) still queued, waiting for the active batch to finish: {remaining}")
        elif not jobs:
            print("Nothing pending or queued.")

    if still_pending:
        print(f"\n{still_pending} job(s) still processing — run this script again later to check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
