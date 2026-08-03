import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path

from src.ingestion.chunkers.hierarchical_chunker import hierarchical_chunker
from src.ingestion.chunkers.sec_section_splitter import split_sec_sections
from src.ingestion.chunkers.semantic_chunker import chunk_with_context
from src.ingestion.chunkers.table_describer import describe_table
from src.ingestion.document_loader import load_document
from src.ingestion.models import Chunk, ContentType, IngestionItem, IngestResult
from src.ingestion.multimodal.table_extractor import extract_tables_from_pdf
from src.ingestion.sec_metadata import SECMetadata

try:
    from src.ingestion.multimodal.image_extractor import (
        extract_images_from_pdf,  # noqa: F401 — availability check; not yet wired into ingest()
    )
    from src.ingestion.multimodal.vision_processor import describe_image, render_pdf_pages  # noqa: F401 — same
    _MULTIMODAL_AVAILABLE = True
except ImportError:
    logging.warning("Multimodal dependencies not found. Image extraction and processing will be unavailable.")
    _MULTIMODAL_AVAILABLE = False

logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = {".png" , ".jpg" , ".jpeg" , ".webp"}
# SEC page-footers use pipe-delimited "Company | Form | Page N" text;
# real prose never contains a literal pipe
_JUNK_CAPTION_PATTERN = re.compile(r"\|")
_TABLE_DESCRIBE_MAX_WORKERS = 5


def _build_table_content(doc: IngestionItem, sec_metadata: SECMetadata | None = None) -> str:
    section = doc.metadata.get("section", "")
    caption = doc.metadata.get("table_caption", "")
    if caption and _JUNK_CAPTION_PATTERN.search(caption):
        caption = ""

    fiscal_period = sec_metadata.fiscal_period if sec_metadata else ""
    fiscal_year = str(sec_metadata.fiscal_year) if sec_metadata else ""
    description = describe_table(section, caption, doc.content, fiscal_period, fiscal_year)

    parts = [p for p in [section, caption, description] if p]
    parts.append(doc.content)
    return "\n\n".join(parts)

class ChunkingStrategy(StrEnum):
    HIERARCHICAL = "hierarchical"
    CONTEXTUAL = "contextual"

class IngestionPipeline:
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
        chunk_overlap: int = 20,
        render_pdf_pages_as_images: bool = False,
        min_image_area_px: int = 5000,
    ) -> None:
        self.strategy = strategy
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.render_pdf_pages_as_images = render_pdf_pages_as_images
        self.min_image_area_px = min_image_area_px
    
    def ingest(self, file_path: str | Path, sec_metadata: SECMetadata | None = None) -> IngestResult:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        all_chunks: list[Chunk] = []
        errors: list[str] = []

        try:
            raw_docs = load_document(file_path)
        except Exception as e:
            errors.append(f"Failed to load document: {e}")
            return IngestResult(chunks=[], errors=errors)

        if sec_metadata is not None:
            try:
                raw_docs = split_sec_sections(raw_docs, filing_type=sec_metadata.filing_type)
            except Exception as e:
                errors.append(f"Section splitting failed, falling back to unsplit document: {e}")

        text_docs = [d for d in raw_docs if d.content_type == ContentType.TEXT]
        table_docs = [d for d in raw_docs if d.content_type == ContentType.TABLE]

        text_chunks, chunk_errors = self._chunk_docs(text_docs)
        all_chunks.extend(text_chunks)
        errors.extend(chunk_errors)

        table_chunks, table_errors = self._chunks_from_tables(table_docs, sec_metadata)
        all_chunks.extend(table_chunks)
        errors.extend(table_errors)

        if suffix == ".pdf":
            pdf_chunks, pdf_errors = self._process_pdf(file_path)
            all_chunks.extend(pdf_chunks)
            errors.extend(pdf_errors)

        if sec_metadata is not None:
            base_meta = sec_metadata.to_dict()
            for chunk in all_chunks:
                chunk.metadata = {**base_meta, **chunk.metadata}

        return IngestResult(chunks=all_chunks, errors=errors)

    def _chunk_docs(self, text_docs: list[IngestionItem]) -> tuple[list[Chunk], list[str]]:
        chunks: list[Chunk] = []
        errors: list[str] = []
        for doc in text_docs:
            try:
                if not doc.content.strip():
                    continue
                if self.strategy == ChunkingStrategy.CONTEXTUAL:
                    chunks.extend(chunk_with_context(
                        doc,
                        chunk_size=self.child_chunk_size,
                        overlap=self.chunk_overlap,
                    ))
                elif self.strategy == ChunkingStrategy.HIERARCHICAL:
                    chunks.extend(hierarchical_chunker(
                        doc,
                        parent_size=self.parent_chunk_size,
                        child_size=self.child_chunk_size,
                        overlap=self.chunk_overlap,
                    ))
            except Exception as e:
                errors.append(f"Chunking failed for a section of '{doc.metadata.get('filename', doc.id)}': {e}")
        return chunks, errors


    def _chunks_from_tables(
        self, table_docs: list[IngestionItem], sec_metadata: SECMetadata | None = None,
    ) -> tuple[list[Chunk], list[str]]:
        """One chunk per table. Table descriptions (D-16) are the dominant per-filing LLM
        cost/latency, so they're generated concurrently across tables rather than one at a
        time — same pattern chunk_with_context() already uses for text-chunk summaries."""
        chunks: list[Chunk] = []
        errors: list[str] = []
        docs = [d for d in table_docs if d.content.strip()]

        contents: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=_TABLE_DESCRIBE_MAX_WORKERS) as executor:
            future_to_doc = {executor.submit(_build_table_content, doc, sec_metadata): doc for doc in docs}
            for future, doc in future_to_doc.items():
                try:
                    contents[doc.id] = future.result()
                except Exception as e:
                    errors.append(f"Failed to build table chunk: {e}")

        for doc in docs:
            if doc.id not in contents:
                continue
            chunks.append(Chunk(
                id=doc.id,
                document_id=doc.id,
                parent_chunk_id=None,
                content=contents[doc.id],
                content_type=ContentType.TABLE,
                metadata=dict(doc.metadata),
            ))
        return chunks, errors



    def _process_pdf(self, pdf_path: Path) -> tuple[list[Chunk], list[str]]:
        chunks: list[Chunk] = []
        errors: list[str] = []
        doc_id = _stable_id(pdf_path.read_bytes())

        try:
            tables = extract_tables_from_pdf(pdf_path)
        except Exception as e:
            errors.append(f"PDF table extraction failed entirely: {e}")
            return chunks, errors

        for table in tables:
            if not table["markdown"]:
                continue
            try:
                chunks.append(Chunk(
                    id=table["table_id"],
                    document_id=doc_id,
                    parent_chunk_id=None,
                    content=table["markdown"],
                    content_type=ContentType.TABLE,
                    metadata={
                        "source": table["source"],
                        "page_num": table["page_num"],
                        "filename": pdf_path.name,
                    },
                ))
            except Exception as e:
                errors.append(f"Failed to build PDF table chunk: {e}")

        return chunks, errors
    
def _stable_id(content: bytes, suffix: str = "") -> str:
    key = content + suffix.encode("utf-8") if suffix else content
    return hashlib.sha256(key).hexdigest()[:16]


