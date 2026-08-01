from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.models.factory import get_chat_model


class ExtractedFilters(BaseModel):
    """Structured extraction of which SEC filing(s) a question refers to, if any."""

    tickers: list[str] | None = Field(
        default=None,
        description="Stock ticker symbol(s) (e.g. ['AAPL']) for the company/companies clearly "
        "named or implied. Include multiple tickers for comparative questions naming several "
        "companies (e.g. 'compare Apple and Microsoft' -> ['AAPL', 'MSFT']). None if the "
        "question is general or doesn't name any specific company.",
    )
    filing_type: str | None = Field(
        default=None,
        description="One of '10-K', '10-Q', '8-K' if the question implies a specific filing "
        "type — e.g. 'annual report' or 'yearly' implies '10-K'; 'quarterly' or a specific "
        "quarter (Q1/Q2/Q3) implies '10-Q'. None if unspecified.",
    )
    fiscal_years: list[int] | None = Field(
        default=None,
        description="List of fiscal years referenced, each a 4-digit integer (e.g. 'fiscal "
        "2024' -> [2024]). Include multiple years for comparisons spanning more than one "
        "year, even of the same quarter (e.g. 'compare Q2 2026 to Q2 2025' -> [2026, 2025]). "
        "None if unspecified or relative ('most recent').",
    )
    fiscal_periods: list[str] | None = Field(
        default=None,
        description="List of fiscal periods referenced, each one of 'FY', 'Q1', 'Q2', 'Q3'. "
        "Include multiple periods for comparative questions spanning more than one quarter "
        "(e.g. 'compare Q1 and Q3' -> ['Q1', 'Q3']). Use 'FY' for annual/yearly filings. "
        "None if unspecified.",
    )


def extract_filters_raw(question: str) -> ExtractedFilters | None:
    """Runs the structured extraction and returns the raw result, so callers can inspect
    e.g. how many fiscal years/periods were detected before deciding how to retrieve.
    Fails open: returns None on any error."""
    try:
        model = get_chat_model("query_transform")
        extractor = model.with_structured_output(ExtractedFilters)
        return extractor.invoke(
            f"Extract which company/filing this question refers to, if any: {question}"
        )
    except Exception:
        return None


def build_filter(
    tickers: list[str] | None = None,
    filing_type: str | None = None,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
) -> Filter | None:
    """Builds a Qdrant Filter from explicit values."""
    conditions = []
    if tickers:
        upper_tickers = [t.upper() for t in tickers]
        match = MatchValue(value=upper_tickers[0]) if len(upper_tickers) == 1 else MatchAny(any=upper_tickers)
        conditions.append(FieldCondition(key="ticker", match=match))
    if filing_type:
        conditions.append(FieldCondition(key="filing_type", match=MatchValue(value=filing_type)))
    if fiscal_year:
        conditions.append(FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year)))
    if fiscal_period:
        conditions.append(FieldCondition(key="fiscal_period", match=MatchValue(value=fiscal_period)))

    return Filter(must=conditions) if conditions else None
