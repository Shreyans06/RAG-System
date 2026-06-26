

from src.ingestion.models import IngestionItem , ContentType

from pathlib import Path
import fitz  # PyMuPDF
import hashlib
from typing import Any
from docx import Document

SUPPORTED = {".pdf" , ".docx" , ".txt" , ".md" , ".png" , ".jpg" , ".jpeg" , ".webp"}

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
    
    return _load_image(p)

def _load_pdf(path: Path) -> list[IngestionItem]:
    docs = []
    with fitz.open(path) as pdf:
        total = pdf.page_count
        for page_num , page in enumerate(pdf , start= 1):
             text = page.get_text("text").strip()
             if not text:
                 continue
             docs.append(IngestionItem(
                 id = _doc_id(path , page_num),
                 content = text,
                 content_type = ContentType.TEXT,
                 metadata = _base_meta(path , page = page_num , total_pages = total),
             ))
        return docs

def _load_docx(path: Path) -> list[IngestionItem]:
    doc = Document(path)
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return [IngestionItem(
        id = _doc_id(path),
        content = text,
        content_type = ContentType.TEXT,
        metadata = _base_meta(path),
    )]

def _load_text(path: Path) -> list[IngestionItem]:
    text = path.read_text(encoding="utf-8" , errors="replace").strip()
    return [IngestionItem(
        id = _doc_id(path),
        content = text,
        content_type = ContentType.TEXT,
        metadata = _base_meta(path),
    )]

def _load_image(path: Path) -> list[IngestionItem]:
    return [IngestionItem(
        id = _doc_id(path),
        content = "",
        content_type = ContentType.IMAGE,
        metadata = {**_base_meta(path) , "image_path": str(path)}, 
    )]

def _doc_id(source: Path , page: int | None = None) -> str:
    key = f"{source} : {page}" if page is not None else str(source)
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def _base_meta(path: Path , **extra: Any) -> dict[str , Any]:
    return {
        "filename": path.name,
        "filepath": str(path),
        **extra
    }


