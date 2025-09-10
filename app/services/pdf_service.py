"""
pdf_service.py
Service for PDF ingestion, embedding, and Qdrant upload.
"""

from typing import List, Optional
from app.utils.pdf_utils import PDFUtils
from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
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
        """
        Ingests a PDF file, extracts text, computes embeddings, and uploads to Qdrant.
        Args:
            pdf_path (str): Path to the PDF file.
            metadata (dict): Optional metadata to store with each page.
        """
        try:
            # Extract text content from PDF pages
            pages_text = PDFUtils.extract_text_from_pages(pdf_path)
            logger.info(f"Processing {len(pages_text)} pages from {metadata.get('filename', pdf_path)}")
            
            # Filter out empty pages
            non_empty_pages = [(idx, text) for idx, text in enumerate(pages_text) if text.strip()]
            
            if not non_empty_pages:
                logger.warning("No text content found in PDF")
                return
            
            # Prepare texts and payloads for batch processing
            texts = [text for _, text in non_empty_pages]
            payloads = []
            
            for idx, text in non_empty_pages:
                payload = metadata.copy() if metadata else {}
                payload.update({
                    "page": idx,
                    "text": text,
                    "page_number": idx + 1,
                    "document_type": "pdf"
                })
                payloads.append(payload)
            
            # Generate embeddings using the hybrid approach
            logger.info("Generating embeddings...")
            dense_vectors = self.embedder.embed_dense(texts)
            sparse_vectors = self.embedder.embed_sparse(texts)
            colbert_vectors = self.embedder.embed_colbert(texts)
            
            # Upload to Qdrant using hybrid batch upsert
            self.qdrant.upsert_hybrid_batch(dense_vectors, sparse_vectors, colbert_vectors, payloads)
            
            logger.info(f"Successfully ingested {len(non_empty_pages)} pages with content")
            
        except Exception as e:
            logger.error(f"Error during PDF ingestion: {e}")
            raise
