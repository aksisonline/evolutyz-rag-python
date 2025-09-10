
## RAG Pipeline Steps

## 1. **Data Ingestion**

* User uploads a CSV file with any schema via frontend UI.
* Backend parses each row, dynamically mapping all columns to a Python dictionary.
* Select text columns for ColBERT embedding (as configured); concatenate as needed.
* ColBERT processes selected text; outputs multivector (per-token) embedding.
* Other columns are stored in Qdrant as dynamic payload (metadata JSON).
* Batch upserts multivector and payload for each row into Qdrant’s vector store.

---

## 2. **Query Embedding**

* Chat user sends a question from the Next.js chat interface.
* Backend receives query and applies ColBERT multivector embedding to it.

---

## 3. **Retrieval & Late Interaction Reranking**

* Qdrant performs fast initial semantic search with dense vectors.
* Late interaction: Qdrant applies ColBERT’s token-level similarity scoring between query and document embeddings.
* Optionally, payload metadata filters apply (e.g., filter by category, tags, or other structured data).

---

## 4. **Generation**

* Backend selects top-k reranked documents/snippets.
* Aggregates their content and sources, builds prompt.
* Sends prompt/context to Gemini 2.5 Flash LLM.
* Receives synthesized answer, reasoning, and source citations.

---

## 5. **Evaluation**

* Automated: Faithfulness, semantic similarity, and coverage scoring between generated answer and retrieved context.
* Manual: (Optional) Logging of token-level provenance for later annotation or review in UI.

---

## 6. **UI Response & Display**

* Frontend streams LLM’s answer with attached sources via shadcn AI chatbot block.
* Reasoning and provenance available in chat UI.
* Scroll and typing/streaming indicators managed as in shadcn block logic.

---

## 7. **Monitoring & Ops**

* Backend logs ingestion and query events; displays ingestion status in upload page UI.
* Regular snapshots and backup of Qdrant for DR.
* Health checks and scaling via sharding/replication.

---

## High-Level Logical Diagram

1. **Ingestion:** CSV → Python backend → ColBERT embedding + payload → Qdrant batch insert
2. **Query:** User question → ColBERT query embedding → Qdrant retrieval + reranking → top docs
3. **Augmentation:** Top docs → context prompt → Gemini LLM → generated answer/reasoning/sources
4. **Response:** AI response streaming to UI → source/reasoning display
5. **Evaluation:** Automated + manual metrics → stored in backend/logs
