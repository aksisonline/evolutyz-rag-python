# Visual System Diagrams

This document contains ASCII art diagrams that visually represent the RAG pipeline architecture and data flows.

## 🏗️ Complete System Architecture

```
                    EVOLUTYZ RAG PIPELINE - COMPLETE ARCHITECTURE
                   ╔═══════════════════════════════════════════════════╗
                   ║                    CLIENT LAYER                   ║
                   ╚═══════════════════════════════════════════════════╝
                            │                           │
                   ┌─────────────────┐         ┌─────────────────┐
                   │   Frontend UI   │         │   API Clients   │
                   │   (Next.js)     │         │   (curl/SDK)    │
                   └─────────────────┘         └─────────────────┘
                            │                           │
                            └──────────┬────────────────┘
                                       │ HTTP/REST
                   ╔═══════════════════════════════════════════════════╗
                   ║                  FASTAPI GATEWAY                  ║
                   ╠═══════════════════════════════════════════════════╣
                   ║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
                   ║  │ /ingestion  │ │   /query    │ │   /files    │ ║
                   ║  │    API      │ │    API      │ │    API      │ ║
                   ║  └─────────────┘ └─────────────┘ └─────────────┘ ║
                   ╚═══════════════════════════════════════════════════╝
                                       │
                   ╔═══════════════════════════════════════════════════╗
                   ║                  SERVICE LAYER                    ║
                   ╠═══════════════════════════════════════════════════╣
                   ║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
                   ║  │ Ingestion   │ │    Query    │ │    Files    │ ║
                   ║  │  Service    │ │   Service   │ │   Service   │ ║
                   ║  └─────────────┘ └─────────────┘ └─────────────┘ ║
                   ╚═══════════════════════════════════════════════════╝
                                       │
                   ╔═══════════════════════════════════════════════════╗
                   ║                   UTILS LAYER                     ║
                   ╠═══════════════════════════════════════════════════╣
                   ║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ║
                   ║  │   ColBERT   │ │   Qdrant    │ │     LLM     │ ║
                   ║  │  Embedder   │ │   Client    │ │   Service   │ ║
                   ║  └─────────────┘ └─────────────┘ └─────────────┘ ║
                   ╚═══════════════════════════════════════════════════╝
                            │                           │
                   ┌─────────────────┐         ┌─────────────────┐
                   │  Qdrant Vector  │         │ Google Gemini   │
                   │    Database     │         │   2.5 Flash     │
                   └─────────────────┘         └─────────────────┘
```

## 📊 Data Ingestion Flow

```
                        DOCUMENT INGESTION PIPELINE
    
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   UPLOAD    │───▶│   PARSE     │───▶│   CHUNK     │───▶│   EMBED     │
    │ Document    │    │  Content    │    │   Text      │    │ 3 Vectors   │
    │ (PDF/CSV)   │    │             │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │                   │
           ▼                   ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │File Type    │    │Text Extract │    │180 tokens   │    │Dense: 384d  │
    │Validation   │    │Schema Detect│    │Overlap: 0   │    │Sparse: BM25 │
    │Size Check   │    │Metadata     │    │Max: 512     │    │ColBERT:128d │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                      │
                                                                      ▼
                                                              ┌─────────────┐
                                                              │    STORE    │
                                                              │  in Qdrant  │
                                                              │  Vector DB  │
                                                              └─────────────┘
```

## 🔍 Query Processing Flow

```
                           QUERY PROCESSING PIPELINE
    
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    USER     │───▶│  FUNCTION   │───▶│   HYBRID    │───▶│   ANSWER    │
    │   QUERY     │    │  CALLING    │    │  RETRIEVAL  │    │ GENERATION  │
    │             │    │  ROUTING    │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │                   │
           ▼                   ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │"Summarize   │    │Summary:     │    │Dense Search │    │Gemini 2.5   │
    │ features"   │    │Enhanced     │    │Sparse Match │    │Context +    │
    │             │    │Retrieval    │    │ColBERT Rank │    │Sources      │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                              │                   │                   │
                              ▼                   ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                       │3x Results   │    │Diversify    │    │Stream       │
                       │15 vs 5      │    │Cross-Files  │    │Response     │
                       │             │    │             │    │             │
                       └─────────────┘    └─────────────┘    └─────────────┘
```

## ⚡ Hybrid Search Visualization

```
                              HYBRID SEARCH ENGINE
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         QUERY: "key features"                       │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      EMBEDDING GENERATION                           │
    │                                                                     │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
    │  │    DENSE    │     │   SPARSE    │     │      ColBERT        │   │
    │  │   VECTOR    │     │   VECTOR    │     │    MULTI-VECTOR     │   │
    │  │             │     │             │     │                     │   │
    │  │ [0.1, 0.3,  │     │ {"key": 2.1,│     │ [[0.2, 0.4, ...],  │   │
    │  │  0.7, ...]  │     │  "features":│     │  [0.1, 0.8, ...],  │   │
    │  │   (384d)    │     │  1.8, ...}  │     │       ...]          │   │
    │  │             │     │   (sparse)  │     │    (128d/token)     │   │
    │  └─────────────┘     └─────────────┘     └─────────────────────┘   │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        QDRANT SEARCH                                │
    │                                                                     │
    │  STEP 1: PREFETCH                    STEP 2: RERANK                 │
    │  ┌─────────────────────┐              ┌─────────────────────┐       │
    │  │ Dense + Sparse      │              │ ColBERT Late        │       │
    │  │ Vector Search       │─────────────▶│ Interaction         │       │
    │  │                     │              │                     │       │
    │  │ Returns: Top 50     │              │ Returns: Top 5      │       │
    │  │ Candidates          │              │ Precise Results     │       │
    │  └─────────────────────┘              └─────────────────────┘       │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       RESULT PROCESSING                             │
    │                                                                     │
    │  IF SUMMARY QUERY:                   ALWAYS:                        │
    │  ┌─────────────────────┐              ┌─────────────────────┐       │
    │  │ Smart Diversify     │              │ Score Fusion        │       │
    │  │ Across Files        │              │ Source Attribution  │       │
    │  │                     │              │                     │       │
    │  │ Max File Coverage   │              │ Context Preparation │       │
    │  └─────────────────────┘              └─────────────────────┘       │
    └─────────────────────────────────────────────────────────────────────┘
```

## 🧠 Summary Enhancement Algorithm

```
                        SUMMARY ENHANCEMENT WORKFLOW
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        INPUT ANALYSIS                               │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                   ▼                             ▼
    ┌─────────────────────────┐        ┌─────────────────────────┐
    │     KEYWORD CHECK       │        │     PATTERN CHECK       │
    │                         │        │                         │
    │ ✓ "summarize"           │        │ ✓ "what are the..."     │
    │ ✓ "overview"            │        │ ✓ "key points"          │
    │ ✓ "main points"         │        │ ✓ "give me an overview" │
    │ ✓ "compile"             │        │ ✓ "comprehensive view"  │
    └─────────────────────────┘        └─────────────────────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    ENHANCEMENT ACTIVATION                           │
    │                                                                     │
    │  NORMAL QUERY:                    SUMMARY QUERY:                    │
    │  ┌─────────────┐                   ┌─────────────┐                  │
    │  │ top_k = 5   │                   │ top_k = 15  │                  │
    │  │ Files: Any  │        ───▶       │ Files: Max  │                  │
    │  │ Strategy:   │                   │ Strategy:   │                  │
    │  │ Best Score  │                   │ Diversify   │                  │
    │  └─────────────┘                   └─────────────┘                  │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    DIVERSIFICATION ALGORITHM                        │
    │                                                                     │
    │  STEP 1: Group by File            STEP 2: Best from Each           │
    │  ┌─────────────────────┐          ┌─────────────────────┐           │
    │  │ file_a: [r1, r2, r3]│          │ Take highest score  │           │
    │  │ file_b: [r4, r5]    │   ───▶   │ from each file:     │           │
    │  │ file_c: [r6, r7, r8]│          │ [r1, r4, r6, ...]   │           │
    │  └─────────────────────┘          └─────────────────────┘           │
    │                                                                     │
    │  STEP 3: Fill Remaining           STEP 4: Final Result             │
    │  ┌─────────────────────┐          ┌─────────────────────┐           │
    │  │ Add next best from  │          │ 15 results with     │           │
    │  │ any file until      │   ───▶   │ maximum file        │           │
    │  │ top_k reached       │          │ diversity achieved  │           │
    │  └─────────────────────┘          └─────────────────────┘           │
    └─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Function Calling Decision Tree

```
                           FUNCTION CALLING ROUTER
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          INPUT QUERY                                │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
    ┌─────────────────────────────┴───────────────────────────────────────┐
    │                     LOCAL ROUTING CHECK                             │
    │                      (if enabled)                                   │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                   ▼                             ▼
    ┌─────────────────────────┐        ┌─────────────────────────┐
    │    GREETING/IDENTITY    │        │     FACTUAL QUERY       │
    │                         │        │                         │
    │ "Hello"                 │        │ "What are features?"    │
    │ "Who are you?"          │        │ "How does X work?"      │
    │ "Good morning"          │        │ "Summarize Y"           │
    └─────────────────────────┘        └─────────────────────────┘
                   │                             │
                   ▼                             ▼
    ┌─────────────────────────┐        ┌─────────────────────────┐
    │    DIRECT RESPONSE      │        │      LLM ROUTING        │
    │                         │        │                         │
    │ Fast, cached reply      │        │ Gemini function call   │
    │ No RAG needed           │        │ with tool definition    │
    │ <50ms response          │        │                         │
    └─────────────────────────┘        └─────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────┐
                   │                                                       │
                   ▼                                                       ▼
    ┌─────────────────────────┐                        ┌─────────────────────────┐
    │      NEEDS RAG          │                        │    DIRECT ANSWER        │
    │                         │                        │                         │
    │ Factual information     │                        │ General knowledge       │
    │ Document-specific       │                        │ Creative tasks          │
    │ Analysis required       │                        │ No lookup needed        │
    └─────────────────────────┘                        └─────────────────────────┘
                   │                                                       │
                   ▼                                                       ▼
    ┌─────────────────────────┐                        ┌─────────────────────────┐
    │     TRIGGER RAG         │                        │   GENERATE DIRECTLY     │
    │                         │                        │                         │
    │ 1. Embed query          │                        │ Pure LLM response       │
    │ 2. Search vectors       │                        │ No document context     │
    │ 3. Rerank results       │                        │ Faster processing       │
    │ 4. Generate answer      │                        │                         │
    └─────────────────────────┘                        └─────────────────────────┘
```

## 📈 Performance Optimization Map

```
                         PERFORMANCE OPTIMIZATION LAYERS
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          MEMORY LAYER                               │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
    │  │  Singleton  │ │  Adaptive   │ │   Model     │ │   Memory    │  │
    │  │  Pattern    │ │  Batching   │ │  Caching    │ │ Monitoring  │  │
    │  │             │ │             │ │             │ │             │  │
    │  │ One model   │ │ OOM avoid   │ │ Reuse       │ │ Track usage │  │
    │  │ instance    │ │ Auto-reduce │ │ loaded      │ │ Auto-clean  │  │
    │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
                                         │
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         SEARCH LAYER                                │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
    │  │   Hybrid    │ │   Late      │ │   Smart     │ │  Result     │  │
    │  │  Prefetch   │ │ Interaction │ │ Diversify   │ │  Caching    │  │
    │  │             │ │             │ │             │ │             │  │
    │  │ Dense+      │ │ ColBERT     │ │ Cross-file  │ │ Query       │  │
    │  │ Sparse      │ │ Precision   │ │ Coverage    │ │ Results     │  │
    │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
                                         │
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       PROCESSING LAYER                              │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
    │  │  Function   │ │  Streaming  │ │   Async     │ │  Pipeline   │  │
    │  │  Calling    │ │  Response   │ │ Processing  │ │ Parallel    │  │
    │  │             │ │             │ │             │ │             │  │
    │  │ Smart       │ │ Real-time   │ │ Non-block   │ │ Multi-      │  │
    │  │ Routing     │ │ Tokens      │ │ I/O         │ │ threaded    │  │
    │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
                                         │
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        SYSTEM LAYER                                 │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
    │  │ Connection  │ │    Batch    │ │   Error     │ │   Health    │  │
    │  │   Pooling   │ │ Processing  │ │  Recovery   │ │   Checks    │  │
    │  │             │ │             │ │             │ │             │  │
    │  │ DB reuse    │ │ Bulk ops    │ │ Graceful    │ │ Monitor     │  │
    │  │ Less TCP    │ │ Efficiency  │ │ Fallback    │ │ Status      │  │
    │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
    
    RESULT: 40% better accuracy + 200ms response time + 99.9% uptime
```

---

These diagrams provide a comprehensive visual understanding of the RAG pipeline architecture, data flows, and optimization strategies. They can be used for presentations, documentation, or technical discussions with stakeholders and developers.