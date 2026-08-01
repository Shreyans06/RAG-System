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
                md = to_markdown(table)
                if not md:
                    continue
                results.append({
                    "table_id" : hashlib.sha256(f"{md}:p{page_number}:t{t_idx}".encode()).hexdigest()[:8],
                    "page_num" : page_number,
                    "markdown" : md,
                    "source" : str(pdf_path),
                })
    return results

def to_markdown(table: list[list[str | None]]) -> str:
    def clean(cell: str | None) -> str:
        return (cell or "").strip().replace("|", r"\|")

    rows = [[clean(c) for c in row] for row in table]

    # SEC/EDGAR-generated tables commonly start with one or more fully-blank leading rows
    # (column-width spacers) and include fully-blank columns (visual-alignment spacers) that
    # carry no data — assuming table[0] is a clean header drops these tables entirely.
    header_idx = next((i for i, row in enumerate(rows) if any(row)), None)
    if header_idx is None:
        return ""
    rows = rows[header_idx:]

    max_cols = max(len(row) for row in rows)
    padded = [row + [""] * (max_cols - len(row)) for row in rows]
    keep_cols = [c for c in range(max_cols) if any(row[c] for row in padded)]
    if not keep_cols:
        return ""
    trimmed = [[row[c] for c in keep_cols] for row in padded]

    header, *body = trimmed
    separator = ["-----"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines)

