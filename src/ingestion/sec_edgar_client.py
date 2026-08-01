import json
import time
from pathlib import Path

import httpx

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"

_MIN_REQUEST_INTERVAL = 0.15  # ~6-7 req/s, comfortably under SEC's 10 req/s fair-access limit


def _infer_fiscal_year_and_period(filing_type: str, report_date: str, fiscal_year_end: str | None) -> tuple[int, str]:
    report_year = int(report_date[:4])
    report_month = int(report_date[5:7])
    report_day = int(report_date[8:10])

    # 52/53-week fiscal calendars (Apple's included) sometimes push a quarter-end date
    # a few days into the next calendar month (e.g. FY2023 Q3 ended July 1st, not June 30th).
    # Treat the first few days of a month as still belonging to the prior month so the
    # month-bucket math below doesn't misclassify the quarter.
    effective_year, effective_month = report_year, report_month
    if report_day <= 5:
        effective_month -= 1
        if effective_month == 0:
            effective_month = 12
            effective_year -= 1

    fy_end_month = int(fiscal_year_end[:2]) if fiscal_year_end and len(fiscal_year_end) >= 2 else 12
    fiscal_year = effective_year if effective_month <= fy_end_month else effective_year + 1

    if filing_type == "10-K":
        return fiscal_year, "FY"
    if filing_type == "10-Q":
        fy_start_month = (fy_end_month % 12) + 1
        months_since_start = (effective_month - fy_start_month) % 12
        quarter = months_since_start // 3 + 1
        return fiscal_year, f"Q{quarter}"
    return fiscal_year, "N/A"




class SECEdgarClient:
    def __init__(self, user_agent: str, cache_dir: str | Path = "data/raw"):
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent containing a real contact email, e.g. "
                "'Your Name your.email@example.com'"
            )
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0)
        self._last_request_time = 0.0
        self._ticker_to_cik: dict[str, str] | None = None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get_json(self, url: str) -> dict:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response.json()

    def _load_ticker_map(self) -> dict[str, str]:
        cache_path = self.cache_dir / "company_tickers.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
        else:
            raw = self._get_json(_TICKERS_URL)
            cache_path.write_text(json.dumps(raw))
        return {entry["ticker"].upper(): f"{entry['cik_str']:010d}" for entry in raw.values()}

    def get_cik(self, ticker: str) -> str:
        if self._ticker_to_cik is None:
            self._ticker_to_cik = self._load_ticker_map()
        cik = self._ticker_to_cik.get(ticker.upper())
        if cik is None:
            raise ValueError(f"Unknown ticker: {ticker}")
        return cik

    def get_filings(self, ticker: str, filing_types: list[str], years: list[int] | None = None) -> list[dict]:
        cik = self.get_cik(ticker)
        submissions = self._get_json(_SUBMISSIONS_URL.format(cik=cik))
        fiscal_year_end = submissions.get("fiscalYearEnd")

        # "recent" only holds the ~1000 most-recent filings of ANY form type for this CIK.
        # Companies that file frequently (8-Ks, proxies, etc.) push older 10-K/10-Qs out of
        # that window into separate paginated pages listed under filings.files — fetch those
        # too, or older filings silently go missing with no error.
        pages = [submissions["filings"]["recent"]]
        for older in submissions["filings"].get("files", []):
            pages.append(self._get_json(f"https://data.sec.gov/submissions/{older['name']}"))

        results = []
        for page in pages:
            for i, form in enumerate(page["form"]):
                if form not in filing_types:
                    continue
                report_date = page["reportDate"][i]
                fiscal_year, fiscal_period = _infer_fiscal_year_and_period(form, report_date, fiscal_year_end)
                if years is not None and fiscal_year not in years:
                    continue
                results.append(
                    {
                        "ticker": ticker.upper(),
                        "cik": cik,
                        "company": submissions.get("name", ticker.upper()),
                        "filing_type": form,
                        "filing_date": page["filingDate"][i],
                        "fiscal_year": fiscal_year,
                        "fiscal_period": fiscal_period,
                        "accession_number": page["accessionNumber"][i],
                        "primary_document": page["primaryDocument"][i],
                    }
                )
        return results


    def download_filing(self, filing: dict) -> Path:
        dest = self.cache_dir / filing["ticker"] / filing["filing_type"] / f"{filing['accession_number']}.htm"
        if dest.exists():
            return dest

        accession_nodash = filing["accession_number"].replace("-", "")
        url = _ARCHIVE_URL.format(
            cik_int=int(filing["cik"]), accession_nodash=accession_nodash, primary_doc=filing["primary_document"]
        )
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return dest
