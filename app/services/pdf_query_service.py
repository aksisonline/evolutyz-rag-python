"""
pdf_query_service.py
Service for querying PDF embeddings in Qdrant using ColPali/ColQwen2.
"""

from qdrant_client import QdrantClient, models
import torch
import os
import logging

logger = logging.getLogger(__name__)

class PDFQueryService:
    """Service for querying PDF embeddings in Qdrant with lazy model loading."""
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.model = None
        self.processor = None
        logger.info("PDFQueryService initialized (models will load on first use)")
    
    def _load_models(self):
        """Lazy loading of ColPali models"""
        if self.model is None:
            try:
                logger.info("Loading ColPali model (this may take a while)...")
                from colpali_engine.models import ColPali, ColPaliProcessor
                
                self.model = ColPali.from_pretrained(
                    "vidore/colpali-v1.3",
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                ).eval()
                
                self.processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
                logger.info("ColPali model loaded successfully!")
                
            except Exception as e:
                logger.error(f"Failed to load ColPali model: {e}")
                raise

    def query_pdf(self, query: str, search_limit: int = 10, prefetch_limit: int = 100):
        # Load models on first use
        if self.model is None:
            self._load_models()
            
        processed_queries = self.processor.process_queries([query]).to(self.model.device)
        query_embedding = self.model(**processed_queries)[0]
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    limit=prefetch_limit,
                    using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query_embedding,
                    limit=prefetch_limit,
                    using="mean_pooling_rows"
                ),
            ],
            limit=search_limit,
            with_payload=True,
            with_vector=False,
            using="original"
        )
        return response
