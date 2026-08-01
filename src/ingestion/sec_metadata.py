from dataclasses import dataclass, replace
from typing import Any

SEC_COMPANY_KEY = "company"
SEC_TICKER_KEY = "ticker"
SEC_CIK_KEY = "cik"
SEC_FILING_TYPE_KEY = "filing_type"
SEC_FILING_DATE_KEY = "filing_date"
SEC_FISCAL_YEAR_KEY = "fiscal_year"
SEC_FISCAL_PERIOD_KEY = "fiscal_period"
SEC_SECTION_KEY = "section"
SEC_ACCESSION_NUMBER_KEY = "accession_number"
SEC_SOURCE_URL_KEY = "source_url"

@dataclass
class SECMetadata:
    company: str
    ticker: str
    cik: str
    filing_type: str # "10-K" | "10-Q" | "8-K"
    filing_date: str # ISO format
    fiscal_year: int
    fiscal_period: str # "FY" | "Q1" | "Q2" | "Q3"
    accession_number: str
    source_url: str
    section: str | None = None

    def to_dict(self) -> dict[str , Any]:
        raw = {
            SEC_COMPANY_KEY: self.company,
            SEC_TICKER_KEY: self.ticker,
            SEC_CIK_KEY: self.cik,
            SEC_FILING_TYPE_KEY: self.filing_type,
            SEC_FISCAL_YEAR_KEY: self.fiscal_year,
            SEC_FISCAL_PERIOD_KEY: self.fiscal_period,
            SEC_ACCESSION_NUMBER_KEY: self.accession_number,
            SEC_SOURCE_URL_KEY: self.source_url,
            SEC_SECTION_KEY: self.section
        }
        return {k: v for k , v in raw.items() if v is not None}

    def with_section(self , section: str) -> "SECMetadata":
        return replace(self , section = section)
    