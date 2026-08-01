import re

CITATION_PATTERN = re.compile(r"\[([0-9a-f]{8})\]")

def extract_citations(response: str) -> list[str]:
    """Extract citation IDs from a response string."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(response):
        if match not in seen:
            seen.append(match)
    return seen

def citation_precision(response: str , supplied_chunk_ids: list[str]) -> float:
    """Compute citation precision: fraction of cited chunks that were actually retrieved."""
    cited_chunk_ids = extract_citations(response)
    if not cited_chunk_ids:
        return 1.0  # No citations, so precision is perfect
    retrieved_set = set(supplied_chunk_ids)
    cited_set = set(cited_chunk_ids)
    true_positives = len(retrieved_set.intersection(cited_set))
    return true_positives / len(cited_set)

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
    precision = citation_precision(response , supplied_chunk_ids)
    coverage = citation_coverage(response , supplied_chunk_ids)
    return {
        "citation_precision": precision,
        "citation_coverage": coverage,
    }