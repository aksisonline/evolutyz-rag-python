import os
import math
import pandas as pd
from fastapi import UploadFile
from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
from app.models.ingestion import IngestionResponse
from app.utils.segmentation import segment_text_by_tokens, dynamic_segment_text
from app.utils.logging_config import logger

class IngestionService:
    def __init__(self):
        self.qdrant = QdrantClientWrapper()
        self.embedder = ColBERTEmbedder()


    def ingest_csv(self, file: UploadFile) -> IngestionResponse:
        """Ingest a CSV file in memory-safe adaptive batches.

        Strategy:
        - Stream CSV with pandas chunksize (INGEST_BATCH_SIZE env or default 64)
        - For each chunk, further adapt batch size on-the-fly if embedding triggers OOM / allocation errors.
        - ColBERT often the memory bottleneck; we progressively halve the batch until success (min 1).
        - Logs progress every processed batch.
        """
        # Batch & segmentation params (all inside method scope)
        base_batch_size = int(os.getenv("INGEST_BATCH_SIZE", "64"))
        target_tokens = int(os.getenv("CSV_CHUNK_TOKENS", "180"))
        soft_max_tokens = int(os.getenv("CSV_CHUNK_MAX_TOKENS", "300"))
        overlap_tokens = int(os.getenv("CSV_CHUNK_OVERLAP_TOKENS", "0"))
        hard_max_tokens = int(os.getenv("CSV_CHUNK_HARD_MAX_TOKENS", "512"))
        dynamic_enabled = os.getenv("CSV_DYNAMIC_SEGMENT") is not None
        dynamic_target_segments = int(os.getenv("CSV_DYNAMIC_TARGET_SEGMENTS", "12"))
        dynamic_min_tokens = int(os.getenv("CSV_DYNAMIC_MIN_TOKENS", "120"))
        dynamic_max_tokens = int(os.getenv("CSV_DYNAMIC_MAX_TOKENS", str(soft_max_tokens)))
        adaptive_min_batch = 1
        store_text = os.getenv("STORE_CHUNK_TEXT") is not None
        chunk_text_field = os.getenv("CHUNK_TEXT_FIELD", "text")
        chunk_text_max = int(os.getenv("CHUNK_TEXT_MAX_CHARS", "1600"))

        processed_rows = 0
        total_segments = 0
        logger.info(
            f"Starting CSV ingestion (base_batch_size={base_batch_size}, dynamic={dynamic_enabled}) into collection '{self.qdrant.collection_name}'"
        )

        try:
            # Ensure file pointer at start
            try:
                file.file.seek(0)
            except Exception:
                pass

            chunk_iter = pd.read_csv(file.file, chunksize=base_batch_size)
            for chunk_idx, df_chunk in enumerate(chunk_iter):
                base_texts = self._select_text_columns(df_chunk)
                row_payloads = df_chunk.to_dict(orient="records")

                texts: list[str] = []
                payloads: list[dict] = []
                for row_offset, (text, row_payload) in enumerate(zip(base_texts, row_payloads)):
                    if dynamic_enabled:
                        segs = dynamic_segment_text(
                            text,
                            target_segment_count=dynamic_target_segments,
                            min_tokens=dynamic_min_tokens,
                            max_tokens=dynamic_max_tokens,
                            hard_max_tokens=hard_max_tokens,
                            overlap_tokens=overlap_tokens,
                        ) or [text]
                    else:
                        segs = segment_text_by_tokens(
                            text,
                            target_tokens=target_tokens,
                            soft_max_tokens=soft_max_tokens,
                            overlap_tokens=overlap_tokens,
                            hard_max_tokens=hard_max_tokens,
                        ) or [text]
                    for seg_idx, seg in enumerate(segs):
                        seg_payload = dict(row_payload)
                        seg_payload.update(
                            {
                                "_segment_index": seg_idx,
                                "_segments_total": len(segs),
                                "_original_row_index": processed_rows + row_offset,
                            }
                        )
                        if store_text:
                            seg_payload[chunk_text_field] = seg if len(seg) <= chunk_text_max else seg[:chunk_text_max] + "…"
                        texts.append(seg)
                        payloads.append(seg_payload)
                total_segments += len(texts)

                # Adaptive embedding/upsert
                start = 0
                current_batch_size = min(len(texts), base_batch_size)
                while start < len(texts):
                    end = min(start + current_batch_size, len(texts))
                    sub_texts = texts[start:end]
                    sub_payloads = payloads[start:end]
                    while True:
                        try:
                            dense_vecs = self.embedder.embed_dense(sub_texts)
                            sparse_vecs = self.embedder.embed_sparse(sub_texts)
                            colbert_vecs = self.embedder.embed_colbert(sub_texts)
                            self.qdrant.upsert_hybrid_batch(dense_vecs, sparse_vecs, colbert_vecs, sub_payloads)
                            logger.info(
                                f"Ingestion progress: rows={processed_rows + len(df_chunk)} segments={total_segments} (chunk {chunk_idx}, seg_batch {start}-{end})"
                            )
                            break
                        except Exception as e:
                            msg = str(e).lower()
                            oom_like = any(
                                t in msg for t in ["failed to allocate", "out of memory", "cuda error", "oom", "allocation failed"]
                            )
                            if oom_like and current_batch_size > adaptive_min_batch:
                                new_size = max(adaptive_min_batch, current_batch_size // 2)
                                if new_size == current_batch_size and new_size > adaptive_min_batch:
                                    new_size = adaptive_min_batch
                                logger.warning(
                                    f"Memory issue embedding CSV batch (size={current_batch_size}). Reducing to {new_size}. Error: {e}"
                                )
                                current_batch_size = new_size
                                continue
                            raise
                    start = end
                    if current_batch_size < base_batch_size:
                        current_batch_size = min(base_batch_size, current_batch_size * 2)

                processed_rows += len(df_chunk)

            logger.info(
                f"Ingestion completed successfully. rows={processed_rows} segments={total_segments} dynamic={dynamic_enabled}"
            )
            return IngestionResponse(
                success=True,
                message="Ingestion successful",
                ingested_rows=processed_rows,
                segments=total_segments,
            )
        except Exception as e:
            logger.exception("Ingestion failed")
            return IngestionResponse(
                success=False,
                message="Ingestion failed",
                ingested_rows=processed_rows,
                segments=total_segments or None,
                errors=[str(e)],
            )

    def _select_text_columns(self, df: pd.DataFrame):
        # Select all object (string) columns for embedding
        text_cols = df.select_dtypes(include=["object"]).columns
        return df[text_cols].astype(str).agg(" ".join, axis=1).tolist()

    # Legacy char-based _segment_text removed (token-based segmentation now centralized in segmentation.py)
