
## 1. Stack Overview

**Frontend (Next.js):**

* Chat Page: shadcn/ui AI Chatbot block – streaming, model select, reason/source display, robust message and scroll management.[shadcn](https://www.shadcn.io/blocks/ai-chatbot)
* CSV Upload Page: Upload any schema CSV, view status/progress.

**Backend (Python, FastAPI):**

* Qdrant vector DB (leveraging all features: ColBERT multivector, late-interaction, payload, scaling).
* ColBERT for embeddings and retrieval/reranking.
* Gemini 2.5 Flash API for answer generation.
* Automated evaluation module (faithfulness, similarity, coverage).
* Ops: logging, monitoring, batch handling, backups, scaling. No external notification/webhook.

---

## 2. RAG Pipeline Steps & Logic Flow

## Step 1: Data Ingestion

* User uploads CSV via UI.
* Backend parses each row, dynamically creates payload from all columns (fully flexible schema).
* Selects/concatenates text columns for ColBERT embedding.
* ColBERT generates multivector per row.
* Qdrant upserts: vector (multivector) + payload metadata per data point (batch ops).

## Step 2: Query Processing

* User asks question in chat UI.
* Backend encodes query as multivector via ColBERT.

## Step 3: Retrieval & Reranking

* Qdrant performs fast ANN retrieval (dense similarity).
* Applies late-interaction reranking using ColBERT (token-level semantic match).
* Supports arbitrary metadata filtering via payload.
* Returns top-k ranked items + all payload metadata.

## Step 4: Context Augmentation & Generation

* Backend compiles retrieved contexts, aggregates relevant payload fields.
* Builds augmented prompt for Gemini 2.5 Flash.
* LLM generates answer, reasoning, source citations.

## Step 5: Evaluation

* Automated: Scores for faithfulness (QA/NLI), semantic similarity, coverage.
* Manual: (If used) logs/provenance for annotator review.

## Step 6: Streaming Response

* Backend streams answer, reasoning, sources to chat UI.
* UI uses shadcn block for optimized scroll, transparent reasoning, source attribution.

## Step 7: Ops & Monitoring

* Logs for ingestion/query.
* Batch upload status/progress visible in CSV page.
* Scheduled backups/snapshots; sharding/replication for scaling.

---

## 3. Qdrant Feature Utilization

| Feature               | Usage in Pipeline                            |
| --------------------- | -------------------------------------------- |
| Multivector Embedding | ColBERT support for rich token-level ranking |
| Late Interaction API  | High-precision reranking                     |
| Dynamic Payload       | Per-row metadata, schema-less                |
| ANN & Hybrid Search   | Fast semantic & attribute filtering          |
| Batch Upserts         | Efficient, large-scale ingestion             |
| Sharding/Replication  | Scaling and resilience                       |
| Monitoring            | Log/query metrics, ingestion records         |
| Backups/Snapshots     | Automated DR                                 |
| REST/gRPC APIs        | Async ops, extensible integration            |
| Security/Logging      | API keys, audit trails, error management     |

---

## 4. Frontend Key Features

* **Chat:** Streaming responses, typing indicators, source display, reasoning.
* **CSV Upload:** Any schema, progress+status, summary page.

---

## 5. User Journey

* Data team uploads any CSV: system adapts, ingests, reflects schema in metadata and search.
* End user queries via chat: Precise, context-grounded answers with full provenance and transparency.
* Ops/dev: Scaling, monitoring, backups, all managed server-side.
