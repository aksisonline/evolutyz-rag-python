
import os
import time
from typing import List
from qdrant_client import QdrantClient, models
from app.utils.logging_config import logger
try:
    import httpx  # for exception types
except Exception:  # pragma: no cover
    httpx = None

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
        """Upsert a batch of hybrid (dense+sparse+ColBERT) vectors with safe sub-batching & retries.

        Environment variables:
        - QDRANT_MAX_POINTS_PER_UPSERT (int, default 16)
        - QDRANT_UPSERT_RETRIES (int, default 3)
        - QDRANT_UPSERT_BACKOFF_BASE (float seconds, default 0.5)
        - QDRANT_DISABLE_COLBERT (bool flag) -> if set, omit ColBERT vectors
        """
        from qdrant_client.models import PointStruct
        import uuid

        max_points = int(os.getenv("QDRANT_MAX_POINTS_PER_UPSERT", "16"))
        retries = int(os.getenv("QDRANT_UPSERT_RETRIES", "3"))
        backoff_base = float(os.getenv("QDRANT_UPSERT_BACKOFF_BASE", "0.5"))
        disable_colbert = os.getenv("QDRANT_DISABLE_COLBERT") is not None

        # Build full point list first
        points: List[PointStruct] = []
        for dense, sparse, colbert, payload in zip(dense_vectors, sparse_vectors, colbert_vectors, payloads):
            vector_payload = {
                "all-MiniLM-L6-v2": dense,
                "bm25": sparse.as_object() if hasattr(sparse, 'as_object') else sparse,
            }
            if not disable_colbert:
                vector_payload["colbertv2.0"] = colbert
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_payload,
                payload=payload
            ))

        if not points:
            logger.warning("No points to upsert (empty batch).")
            return

        # Iterate in sub-batches to avoid large payload disconnects
        total = len(points)
        idx = 0
        while idx < total:
            sub = points[idx: idx + max_points]
            attempt = 0
            while True:
                try:
                    self.client.upsert(collection_name=self.collection_name, points=sub)
                    logger.info(f"Upserted {len(sub)} points ({idx + len(sub)}/{total}) into '{self.collection_name}'.")
                    break
                except Exception as e:  # network / protocol / timeout / server disconnect
                    attempt += 1
                    is_protocol = "disconnected" in str(e).lower() or "protocol" in str(e).lower()
                    transient = is_protocol or (httpx and isinstance(e, (httpx.RemoteProtocolError, httpx.TimeoutException)))
                    if attempt < retries and transient:
                        sleep_time = backoff_base * (2 ** (attempt - 1))
                        logger.warning(f"Transient upsert failure (attempt {attempt}/{retries}) for sub-batch size {len(sub)}: {e}. Backing off {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                        # If repeated failures, shrink sub-batch dynamically
                        if len(sub) > 1 and attempt > 1:
                            new_size = max(1, len(sub) // 2)
                            if new_size < len(sub):
                                logger.warning(f"Reducing sub-batch from {len(sub)} to {new_size} due to repeated failures")
                                sub = sub[:new_size]
                        continue
                    logger.error(f"Failed to upsert hybrid points after {attempt} attempts: {e}")
                    raise
            idx += len(sub)

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
