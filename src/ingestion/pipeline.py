import hashlib
import logging
import tempfile
from enum import Enum
from pathlib import Path

from src.ingestion.chunkers.hierarchical_chunker import hierarchical_chunker
from src.ingestion.chunkers.semantic_chunker import chunk_with_context
from src.ingestion.document_loader import load_document
from src.ingestion.models import Chunk, ContentType, IngestionItem
from src.ingestion.multimodal.table_extractor import extract_tables_from_pdf

try:
    from src.ingestion.multimodal.image_extractor import extract_images_from_pdf
    from src.ingestion.multimodal.vision_processor import describe_image , render_pdf_pages
    _MULTIMODAL_AVAILABLE = True
except ImportError:
    logging.warning("Multimodal dependencies not found. Image extraction and processing will be unavailable.")
    _MULTIMODAL_AVAILABLE = False

logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = {".png" , ".jpg" , ".jpeg" , ".webp"}

class ChunkingStrategy(str , Enum):
    HIERARCHICAL = "hierarchical"
    CONTEXTUAL = "contextual"

class IngestionPipeline:
    def __init__(
        self,
        anthropic_api_key: str | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
        chunk_overlap: int = 20,
        contextual_summary_model: str = "claude-haiku-4-5-20251001",
        render_pdf_pages_as_images: bool = False,
        min_image_area_px: int = 5000,
    ) -> None:
        self.anthropic_api_key = anthropic_api_key
        self.strategy = strategy
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.contextual_summary_model = contextual_summary_model
        self.render_pdf_pages_as_images = render_pdf_pages_as_images
        self.min_image_area_px = min_image_area_px
    
    def ingest(self, file_path: str | Path) -> list[Chunk]:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        all_chunks: list[Chunk] = []

        raw_docs = load_document(file_path)
        text_chunks = self._chunk_docs(raw_docs)
        all_chunks.extend(text_chunks)

        if suffix == ".pdf":
            all_chunks.extend(self._process_pdf(file_path))

        text_count = sum(1 for c in all_chunks if c.content_type ==ContentType.TEXT)
        return all_chunks

    def _chunk_docs(self, raw_docs: list[IngestionItem]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in raw_docs:
            if not doc.content.strip():
                continue
            if self.strategy == ChunkingStrategy.CONTEXTUAL:
                chunks.extend(chunk_with_context(
                    doc,
                    api_key= self.anthropic_api_key if self.anthropic_api_key else "",
                    model= self.contextual_summary_model,
                    chunk_size= self.child_chunk_size,
                    overlap= self.chunk_overlap,
                ))
            elif self.strategy == ChunkingStrategy.HIERARCHICAL:
                chunks.extend(hierarchical_chunker(
                    doc,
                    parent_size= self.parent_chunk_size,
                    child_size= self.child_chunk_size,
                    overlap= self.chunk_overlap,
                ))
        return chunks

    def _process_pdf(self, pdf_path: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        doc_id = _stable_id(pdf_path)

        for table in extract_tables_from_pdf(pdf_path):
            if not table["markdown"]:
                continue
            chunks.append(Chunk(
                id = table["table_id"],
                document_id = doc_id,
                parent_chunk_id = None,
                content = table["markdown"],
                content_type = ContentType.TABLE,
                metadata = {
                    "source" : table["source"],
                    "page_num" : table["page_num"],
                    "filename" : pdf_path.name,
                },
            ))
        
        return chunks
    
def _stable_id(path: Path, suffix: str = "") -> str:
    key = f"{path}:{suffix}" if suffix else str(path)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


