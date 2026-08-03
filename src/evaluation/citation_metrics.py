import re

# Citations are sequential per-response numbers (e.g. [1], [2]), not hashed IDs — small
# integers are far easier for a human to visually scan and tell apart in generated prose
# than random-looking hex, even though hex IDs were already collision-safe (see D-17).
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def extract_citations(response: str) -> list[str]:
    """Extract citation numbers from a response string."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(response):
        if match not in seen:
            seen.append(match)
    return seen

def citation_coverage(response: str , supplied_chunk_ids: list[str]) -> float:
    """Compute citation coverage: fraction of retrieved chunks that were cited."""
    cited_chunk_ids = extract_citations(response)
    if not supplied_chunk_ids:
        return 0.0  # No retrieved chunks, so coverage is zero
    retrieved_set = set(supplied_chunk_ids)
    cited_set = set(cited_chunk_ids)
    true_positives = len(retrieved_set.intersection(cited_set))
    return true_positives / len(retrieved_set)

def compute_citation_metrics(response: str , supplied_chunk_ids: list[str]) -> dict:
    """Compute citation metrics for a given response and the supplied chunk IDs."""
    coverage = citation_coverage(response , supplied_chunk_ids)
    return {
        "citation_coverage": coverage,
    }