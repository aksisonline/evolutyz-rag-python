import pandas as pd
from fastapi import UploadFile
from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
from app.models.ingestion import IngestionResponse
from app.utils.logging_config import logger

class IngestionService:
    def __init__(self):
        self.qdrant = QdrantClientWrapper()
        self.embedder = ColBERTEmbedder()


    def ingest_csv(self, file: UploadFile) -> IngestionResponse:
        try:
            df = pd.read_csv(file.file)
            payloads = df.to_dict(orient="records")
            texts = self._select_text_columns(df)
            logger.info(f"Starting ingestion: {len(df)} rows, collection={self.qdrant.collection_name}")
            dense_vectors = self.embedder.embed_dense(texts)
            sparse_vectors = self.embedder.embed_sparse(texts)
            colbert_vectors = self.embedder.embed_colbert(texts)
            self.qdrant.upsert_hybrid_batch(dense_vectors, sparse_vectors, colbert_vectors, payloads)
            logger.info("Ingestion completed successfully")
            return IngestionResponse(success=True, message="Ingestion successful", ingested_rows=len(df))
        except Exception as e:
            logger.exception("Ingestion failed")
            return IngestionResponse(success=False, message="Ingestion failed", ingested_rows=0, errors=[str(e)])

    def _select_text_columns(self, df: pd.DataFrame):
        # Select all object (string) columns for embedding
        text_cols = df.select_dtypes(include=["object"]).columns
        return df[text_cols].astype(str).agg(" ".join, axis=1).tolist()
