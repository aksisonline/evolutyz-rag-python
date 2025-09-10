
import os
from typing import List, Iterable, Optional
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import logging

logger = logging.getLogger(__name__)

class ColBERTEmbedder:
    _instance: Optional['ColBERTEmbedder'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ColBERTEmbedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ColBERTEmbedder._initialized:
            return
            
        try:
            logger.info("Initializing embedding models...")
            
            dense_model_name = os.getenv("DENSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            colbert_model_name = os.getenv("COLBERT_EMBEDDING_MODEL", "colbert-ir/colbertv2.0")
            sparse_model_name = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
            
            logger.info(f"Loading dense model: {dense_model_name}")
            self.dense_model = TextEmbedding(model_name=dense_model_name)
            
            logger.info(f"Loading sparse model: {sparse_model_name}")
            self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
            
            # Check supported ColBERT models first
            logger.info("Checking supported ColBERT models...")
            supported_models = LateInteractionTextEmbedding.list_supported_models()
            logger.info(f"Supported ColBERT models: {supported_models}")
            
            if colbert_model_name in [model["model"] for model in supported_models]:
                logger.info(f"Loading ColBERT model: {colbert_model_name}")
                self.colbert_model = LateInteractionTextEmbedding(model_name=colbert_model_name)
            else:
                logger.warning(f"Model {colbert_model_name} not supported. Trying default ColBERT model...")
                # Try the first supported model
                if supported_models:
                    default_model = supported_models[0]["model"]
                    logger.info(f"Loading default ColBERT model: {default_model}")
                    self.colbert_model = LateInteractionTextEmbedding(model_name=default_model)
                else:
                    logger.warning("No supported ColBERT models found")
                    self.colbert_model = None
            
            ColBERTEmbedder._initialized = True
            logger.info("All embedding models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            # Fallback to simpler models if ColBERT fails
            try:
                logger.info("Attempting fallback to simpler models...")
                self.dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                self.colbert_model = None  # Disable ColBERT if it fails
                ColBERTEmbedder._initialized = True
                logger.info("Initialized with fallback models (ColBERT disabled)")
            except Exception as fallback_error:
                logger.error(f"Fallback initialization failed: {fallback_error}")
                raise

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        return [vec for vec in self.dense_model.embed(texts)]

    def embed_sparse(self, texts: List[str]):
        return [vec for vec in self.sparse_model.embed(texts)]

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        if self.colbert_model is None:
            logger.warning("ColBERT model not available, returning empty embeddings")
            return [[] for _ in texts]
        return [token_vectors for token_vectors in self.colbert_model.embed(texts)]

    def embed_dense_query(self, query: str) -> List[float]:
        return next(self.dense_model.embed([query]))

    def embed_sparse_query(self, query: str):
        return next(self.sparse_model.query_embed(query))

    def embed_colbert_query(self, query: str) -> List[List[float]]:
        if self.colbert_model is None:
            logger.warning("ColBERT model not available, returning empty query embedding")
            return []
        return next(self.colbert_model.embed([query]))
