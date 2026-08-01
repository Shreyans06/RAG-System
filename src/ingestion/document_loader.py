import hashlib
import re
import warnings
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from docx import Document

from src.ingestion.models import ContentType, IngestionItem
from src.ingestion.multimodal.table_extractor import to_markdown

SUPPORTED = {".pdf" , ".docx" , ".txt" , ".md" , ".png" , ".jpg" , ".jpeg" , ".webp" , ".htm" , ".html"}
_HIDDEN_STYLE_PATTERN = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def load_document(path: str | Path) -> list[IngestionItem]:
    """
    Load documents from a given path. Return a list of IngestionItem objects.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types are: {SUPPORTED}")
    
    if suffix == ".pdf":
        return _load_pdf(p)
    elif suffix == ".docx":
        return _load_docx(p)
    elif suffix in {".txt", ".md"}:
        return _load_text(p)
    elif suffix in {".htm" , ".html"}:
        return _load_html(p)
    
    return _load_image(p)

def _load_html(path: Path) -> list[IngestionItem]:
    html = path.read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script" , "style"]):
        tag.decompose()

    for tag in soup.find_all("ix:header"):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE_PATTERN):
        tag.decompose()

    items: list[IngestionItem] = []
    for t_idx, table_tag in enumerate(soup.find_all("table")):
        grid = _table_to_grid(table_tag)
        if not grid or not any(any(cell for cell in row) for row in grid):
            table_tag.decompose()
            continue
        md = to_markdown(grid)
        if not md:
            table_tag.decompose()
            continue
        caption = _preceding_text(table_tag)
        meta = _base_meta(path , table_caption=caption) if caption else _base_meta(path)
        table_id = _doc_id(md, t_idx)

        items.append(IngestionItem(
            id=table_id,
            content=md,
            content_type=ContentType.TABLE,
            metadata=meta,
        ))
        table_tag.replace_with(f"\n[[TABLE_MARKER:{table_id}]]\n")


    text = soup.get_text("\n", strip=True)
    if text:
        items.insert(0 , IngestionItem(
            id=_doc_id(text),
            content=text,
            content_type=ContentType.TEXT,
            metadata=_base_meta(path),
        ))
    return items

def _table_to_grid(table_tag) -> list[list[str]]:
    return [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["td" , "th"])]
        for row in table_tag.find_all("tr")
    ]

def _preceding_text(tag, max_len: int = 200) -> str:
    """ Nearest non-empty text immediately before the table in DOM."""
    node = tag.find_previous(string=True)
    while node is not None and not node.strip():
        node = node.find_previous(string=True)
    return node.strip()[:max_len] if node else ""

def _load_pdf(path: Path) -> list[IngestionItem]:
    docs = []
    with fitz.open(path) as pdf:
        total = pdf.page_count
        for page_num , page in enumerate(pdf , start= 1):
             text = page.get_text("text").strip()
             if not text:
                 continue
             docs.append(IngestionItem(
                 id = _doc_id(text , page_num),
                 content = text,
                 content_type = ContentType.TEXT,
                 metadata = _base_meta(path , page = page_num , total_pages = total),
             ))
        return docs

def _load_docx(path: Path) -> list[IngestionItem]:
    doc = Document(path)
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return [IngestionItem(
        id = _doc_id(text),
        content = text,
        content_type = ContentType.TEXT,
        metadata = _base_meta(path),
    )]

def _load_text(path: Path) -> list[IngestionItem]:
    text = path.read_text(encoding="utf-8" , errors="replace").strip()
    return [IngestionItem(
        id = _doc_id(text),
        content = text,
        content_type = ContentType.TEXT,
        metadata = _base_meta(path),
    )]

def _load_image(path: Path) -> list[IngestionItem]:
    return [IngestionItem(
        id = _doc_id(path.read_bytes()),
        content = "",
        content_type = ContentType.IMAGE,
        metadata = {**_base_meta(path) , "image_path": str(path)}, 
    )]

def _doc_id(content: str | bytes, extra: str | int | None = None) -> str:
    """Hashes on content, not file path — path-based IDs meant re-ingesting the same file at
    a different location (e.g. a fresh tempfile path on every manual upload) created brand
    new chunk IDs instead of overwriting existing ones, causing duplicate vectors in Qdrant.
    Content hashing makes re-ingestion idempotent: same content -> same ID -> Qdrant upsert
    cleanly overwrites instead of duplicating."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    if extra is not None:
        data += f":{extra}".encode()
    return hashlib.sha256(data).hexdigest()[:16]

def _base_meta(path: Path , **extra: Any) -> dict[str , Any]:
    return {
        "filename": path.name,
        "filepath": str(path),
        **extra
    }


