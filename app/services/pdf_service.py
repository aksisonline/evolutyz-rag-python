"""
pdf_service.py
Service for PDF ingestion, embedding, and Qdrant upload.
"""

from typing import List, Optional
from app.utils.pdf_utils import PDFUtils
from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
from app.utils.segmentation import segment_text_by_tokens, dynamic_segment_text
import os
import logging

logger = logging.getLogger(__name__)

class PDFIngestionService:
    """Service for ingesting and indexing PDF files into Qdrant."""
    _instance: Optional['PDFIngestionService'] = None
    _initialized = False
    
    def __new__(cls, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        if cls._instance is None:
            cls._instance = super(PDFIngestionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        if PDFIngestionService._initialized:
            return
            
        self.qdrant = QdrantClientWrapper()
        self.embedder = ColBERTEmbedder()
        PDFIngestionService._initialized = True
        logger.info("PDFIngestionService initialized")

    def ingest_pdf(self, pdf_path: str, metadata: dict = None):
        """Ingest a PDF with ColBERT-aligned token segmentation & adaptive batching.

        Token Env Config:
        - PDF_CHUNK_TOKENS (default 180)
        - PDF_CHUNK_MAX_TOKENS (soft max, default 300)
        - PDF_CHUNK_OVERLAP_TOKENS (default 0)
        - PDF_CHUNK_HARD_MAX_TOKENS (absolute cap, default 512)
        - PDF_BATCH_SIZE (default 64) -> embedding/upsert batch of segments
        """
        try:
            target_tokens = int(os.getenv("PDF_CHUNK_TOKENS", "180"))
            soft_max_tokens = int(os.getenv("PDF_CHUNK_MAX_TOKENS", "300"))
            overlap_tokens = int(os.getenv("PDF_CHUNK_OVERLAP_TOKENS", "0"))
            hard_max_tokens = int(os.getenv("PDF_CHUNK_HARD_MAX_TOKENS", "512"))
            dynamic_enabled = os.getenv("PDF_DYNAMIC_SEGMENT") is not None
            dynamic_target_segments = int(os.getenv("PDF_DYNAMIC_TARGET_SEGMENTS", "12"))
            dynamic_min_tokens = int(os.getenv("PDF_DYNAMIC_MIN_TOKENS", "120"))
            dynamic_max_tokens = int(os.getenv("PDF_DYNAMIC_MAX_TOKENS", str(soft_max_tokens)))
            batch_size = int(os.getenv("PDF_BATCH_SIZE", "64"))
            adaptive_min_batch = 1
            store_text = os.getenv("STORE_CHUNK_TEXT") is not None
            chunk_text_field = os.getenv("CHUNK_TEXT_FIELD", "text")
            chunk_text_max = int(os.getenv("CHUNK_TEXT_MAX_CHARS", "1600"))

            pages_text = PDFUtils.extract_text_from_pages(pdf_path)
            fname = (metadata or {}).get('filename', os.path.basename(pdf_path))
            logger.info(f"Processing {len(pages_text)} pages from {fname}")

            non_empty_pages = [(idx, text) for idx, text in enumerate(pages_text) if text and text.strip()]
            if not non_empty_pages:
                logger.warning("No text content found in PDF")
                return

            # Build segments
            segments = []
            for page_idx, text in non_empty_pages:
                if dynamic_enabled:
                    page_segments = dynamic_segment_text(
                        text,
                        target_segment_count=dynamic_target_segments,
                        min_tokens=dynamic_min_tokens,
                        max_tokens=dynamic_max_tokens,
                        hard_max_tokens=hard_max_tokens,
                        overlap_tokens=overlap_tokens,
                    ) or [text]
                else:
                    page_segments = segment_text_by_tokens(
                        text,
                        target_tokens=target_tokens,
                        soft_max_tokens=soft_max_tokens,
                        overlap_tokens=overlap_tokens,
                        hard_max_tokens=hard_max_tokens,
                    ) or [text]
                for seg_idx, seg in enumerate(page_segments):
                    payload = (metadata or {}).copy()
                    payload.update({
                        "page": page_idx,
                        "page_number": page_idx + 1,
                        "document_type": "pdf",
                        "_segment_index": seg_idx,
                        "_segments_total": len(page_segments),
                        "filename": fname
                    })
                    if store_text:
                        payload[chunk_text_field] = seg if len(seg) <= chunk_text_max else seg[:chunk_text_max] + "…"
                    segments.append((seg, payload))

            logger.info(f"Segmented PDF into {len(segments)} segments (pages={len(non_empty_pages)})")

            start = 0
            current_batch = min(batch_size, len(segments))
            processed_segments = 0
            while start < len(segments):
                end = min(start + current_batch, len(segments))
                sub_texts = [s for s, _ in segments[start:end]]
                sub_payloads = [p for _, p in segments[start:end]]
                attempt_done = False
                while not attempt_done:
                    try:
                        dense_vectors = self.embedder.embed_dense(sub_texts)
                        sparse_vectors = self.embedder.embed_sparse(sub_texts)
                        colbert_vectors = self.embedder.embed_colbert(sub_texts)
                        self.qdrant.upsert_hybrid_batch(dense_vectors, sparse_vectors, colbert_vectors, sub_payloads)
                        processed_segments += len(sub_texts)
                        logger.info(f"PDF ingestion progress: {processed_segments}/{len(segments)} segments")
                        attempt_done = True
                    except Exception as e:
                        msg = str(e).lower()
                        oom_like = any(term in msg for term in ["failed to allocate", "out of memory", "cuda error", "oom", "allocation failed"])
                        if oom_like and current_batch > adaptive_min_batch:
                            new_size = max(adaptive_min_batch, current_batch // 2)
                            if new_size == current_batch and new_size > adaptive_min_batch:
                                new_size = adaptive_min_batch
                            logger.warning(f"Memory issue embedding PDF batch (size={current_batch}). Reducing to {new_size} and retrying. Error: {e}")
                            current_batch = new_size
                            continue
                        raise
                start = end
                if current_batch < batch_size:
                    current_batch = min(batch_size, current_batch * 2)

            logger.info(f"Successfully ingested PDF: pages={len(non_empty_pages)} segments={len(segments)}")
        except Exception as e:
            logger.error(f"Error during PDF ingestion: {e}")
            raise

    # Legacy character-based segmentation method removed (token-based now primary)
