
import os
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional
from app.utils.logging_config import logger

class QdrantClientWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API")
        )
        self.collection_name = os.getenv("COLLECTION_NAME", "hybrid-search")
        self._ensure_hybrid_collection()

    def _ensure_hybrid_collection(self):
        try:
            existing = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists (vectors: {existing.vectors_count}).")
            # Ensure filename index exists for existing collection
            self._ensure_filename_index()
        except Exception:
            logger.info(f"Creating hybrid collection '{self.collection_name}'.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "all-MiniLM-L6-v2": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    ),
                    "colbertv2.0": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0)
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            logger.info(f"Hybrid collection '{self.collection_name}' created.")
            # Create filename index for new collection
            self._ensure_filename_index()

    def _ensure_filename_index(self):
        """Ensure that the filename field is indexed for efficient filtering."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index on 'filename' field for collection {self.collection_name}")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info(f"Index on 'filename' field already exists for collection {self.collection_name}")
            else:
                logger.warning(f"Could not create index on 'filename' field: {e}")

    def upsert_hybrid_batch(self, dense_vectors, sparse_vectors, colbert_vectors, payloads):
        # Each colbert_vectors[i] is a list of token vectors for document i
        from qdrant_client.models import PointStruct
        import uuid
        points = []
        for i, (dense, sparse, colbert, payload) in enumerate(zip(dense_vectors, sparse_vectors, colbert_vectors, payloads)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),  # Generate unique ID
                vector={
                    "all-MiniLM-L6-v2": dense,
                    "bm25": sparse.as_object() if hasattr(sparse, 'as_object') else sparse,
                    "colbertv2.0": colbert
                },
                payload=payload
            ))
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} hybrid points into '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert hybrid points: {e}")
            raise

    def query_hybrid_with_rerank(self, dense_query, sparse_query, colbert_query, filters, top_k):
        from qdrant_client import models
        prefetch = [
            models.Prefetch(query=dense_query, using="all-MiniLM-L6-v2", limit=20),
            models.Prefetch(query=models.SparseVector(**sparse_query.as_object()), using="bm25", limit=20)
        ]
        
        try:
            # Check if ColBERT query is available and valid
            valid_colbert = False
            colbert_len = None
            try:
                if colbert_query is not None:
                    # Support numpy arrays, lists, tuples
                    if hasattr(colbert_query, 'shape'):
                        # numpy ndarray
                        import numpy as np  # local import to avoid hard dependency elsewhere
                        if isinstance(colbert_query, np.ndarray):
                            colbert_len = colbert_query.shape[0]
                            # Convert to list-of-lists if ndarray for Qdrant
                            if colbert_query.dtype != object:
                                colbert_query = colbert_query.tolist()
                        else:
                            # Fallback length
                            colbert_len = len(colbert_query) if hasattr(colbert_query, '__len__') else None
                    else:
                        colbert_len = len(colbert_query) if hasattr(colbert_query, '__len__') else None
                    valid_colbert = colbert_len is not None and colbert_len > 0
            except Exception as conv_err:
                logger.warning(f"ColBERT query inspection/conversion failed, falling back to dense-only: {conv_err}")
                valid_colbert = False

            logger.debug(f"ColBERT query validity: {valid_colbert}; length: {colbert_len}; type: {type(colbert_query)}")

            if valid_colbert:
                # Use ColBERT reranking if available
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=colbert_query,
                    using="colbertv2.0",
                    limit=top_k,
                    with_payload=True,
                    query_filter=filters if filters else None
                )
                logger.info(f"Hybrid query with ColBERT returned {len(results.points)} results.")
            else:
                # Fallback to dense-only query when ColBERT is not available
                logger.warning("ColBERT query not usable (None or empty), using dense-only search")
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,
                    using="all-MiniLM-L6-v2",
                    limit=top_k,
                    with_payload=True,
                    query_filter=filters if filters else None
                )
                logger.info(f"Dense-only query returned {len(results.points)} results.")
            
            return results.points
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            # Try dense-only fallback on any error
            try:
                logger.info("Attempting dense-only fallback query")
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,
                    using="all-MiniLM-L6-v2",
                    limit=top_k,
                    with_payload=True,
                    query_filter=filters if filters else None
                )
                logger.info(f"Fallback query returned {len(results.points)} results.")
                return results.points
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                raise
