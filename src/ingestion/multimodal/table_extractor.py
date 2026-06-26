import hashlib
from pathlib import Path

import pdfplumber

def extract_tables_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract tables from a PDF and return them as a list of dictonaries
    """
    pdf_path = Path(pdf_path)
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for t_idx, table in enumerate(page.extract_tables()):
                if not table or not table[0]:
                    continue
                md = _to_markdown(table)
                if not md:
                    continue
                results.append({
                    "table_id" : hashlib.sha256(f"{pdf_path}:p{page_number}:t{t_idx}".encode()).hexdigest()[:8],
                    "page_num" : page_number,
                    "markdown" : md,
                    "source" : str(pdf_path),
                })
    return results

def _to_markdown(table: list[list[str | None]]) -> str:
    def clean(cell: str | None) -> str:
        return (cell or "").strip().replace("|" , r"\|")
    
    header = [clean(c) for c in table[0]]
    if not any(header):
        return ""

    rows = [[clean(c) for c in row] for row in table[1:]]
    separator = ["-----"] * len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)
