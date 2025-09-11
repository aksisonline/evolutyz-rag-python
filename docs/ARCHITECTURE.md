# RAG Pipeline Architecture & Technical Deep Dive

## System Architecture Overview

The Evolutyz RAG Python application implements a sophisticated multi-stage Retrieval-Augmented Generation pipeline with hybrid search capabilities and intelligent optimizations.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Layer                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │
│  │   Next.js UI  │  │   API Clients │  │   curl/wget   │            │
│  └───────────────┘  └───────────────┘  └───────────────┘            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI Application                           │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      API Gateway                                │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │ │
│  │  │ /ingestion  │ │   /query    │ │   /files    │               │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Service Layer                               │ │
│  │  ┌──────────────────┐ ┌────────────────┐ ┌───────────────────┐  │ │
│  │  │ IngestionService │ │  QueryService  │ │  FilesService     │  │ │
│  │  └──────────────────┘ └────────────────┘ └───────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Utilities Layer                             │ │
│  │  ┌──────────────────┐ ┌────────────────┐ ┌───────────────────┐  │ │
│  │  │ ColBERTEmbedder  │ │ QdrantClient   │ │ LLMService        │  │ │
│  │  └──────────────────┘ └────────────────┘ └───────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Qdrant Vector Database                       │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐   │ │
│  │  │ Dense        │ │ Sparse       │ │ ColBERT Multivectors  │   │ │
│  │  │ Vectors      │ │ Vectors      │ │ (Late Interaction)    │   │ │
│  │  │ (384d)       │ │ (BM25)       │ │ (128d per token)      │   │ │
│  │  └──────────────┘ └──────────────┘ └────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       External Services                             │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   Google Gemini 2.5 Flash                      │ │
│  │              (Answer Generation & Synthesis)                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow Diagrams

### 1. Document Ingestion Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│   Upload    │    │   Extract    │    │    Generate     │    │   Store    │
│ Document    │───▶│   Content    │───▶│   Embeddings    │───▶│ in Qdrant  │
│ (PDF/CSV)   │    │              │    │                 │    │            │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────┘
                           │                     │
                           ▼                     ▼
                   ┌──────────────┐    ┌─────────────────┐
                   │   Chunk      │    │  Dense Vector   │
                   │  Content     │    │  Sparse Vector  │
                   │              │    │  ColBERT Multi  │
                   └──────────────┘    └─────────────────┘

Detailed Steps:
1. File Upload & Validation
   ├── MIME type checking
   ├── File size validation
   └── Format verification

2. Content Extraction
   ├── CSV: Row-by-row parsing with dynamic schema
   ├── PDF: ColPali visual extraction + text parsing
   └── Metadata preservation

3. Intelligent Chunking
   ├── Token-based segmentation (default: 180 tokens)
   ├── Overlap management (configurable)
   ├── Dynamic chunking (optional)
   └── Hard limits enforcement (512 tokens max)

4. Embedding Generation (Parallel)
   ├── Dense: sentence-transformers/all-MiniLM-L6-v2 (384d)
   ├── Sparse: Qdrant/bm25 (keyword matching)
   └── ColBERT: colbert-ir/colbertv2.0 (128d per token)

5. Qdrant Storage
   ├── Batch processing (default: 64 points)
   ├── Adaptive memory management
   ├── Metadata indexing (filename field)
   └── Multi-vector point creation
```

### 2. Query Processing Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│    User     │    │   Function   │    │     Hybrid      │    │  Answer    │
│   Query     │───▶│   Calling    │───▶│   Retrieval     │───▶│ Generation │
│             │    │   Routing    │    │                 │    │            │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────┘
                           │                     │                     │
                           ▼                     ▼                     ▼
                   ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
                   │   Detect     │    │   Multi-Vector  │    │  Gemini    │
                   │   Intent     │    │   Search +      │    │  2.5 Flash │
                   │              │    │   Reranking     │    │            │
                   └──────────────┘    └─────────────────┘    └────────────┘

Detailed Steps:
1. Query Analysis
   ├── Summary detection (keywords: summarize, overview, etc.)
   ├── Greeting/identity detection
   ├── Factual question identification
   └── Style parameter processing

2. Function Calling Decision
   ├── Local routing (if enabled)
   ├── LLM-based routing (default)
   └── Eager retrieval (if configured)

3. Embedding Generation
   ├── Dense query embedding (384d)
   ├── Sparse query embedding (BM25)
   └── ColBERT query embedding (per-token)

4. Hybrid Retrieval
   ├── Qdrant prefetch (dense + sparse)
   ├── ColBERT reranking (late interaction)
   ├── Score fusion and ranking
   └── Result diversification (for summaries)

5. Context Preparation
   ├── Result aggregation
   ├── Source compilation
   ├── Prompt construction
   └── Token limit management

6. Answer Generation
   ├── Gemini API call (streaming)
   ├── Response formatting
   ├── Source attribution
   └── Metrics compilation
```

### 3. Summary Enhancement Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│   Summary   │    │   Enhanced   │    │  Diversified    │    │ Comprehensive│
│  Detection  │───▶│  Retrieval   │───▶│   Results       │───▶│   Answer     │
│             │    │              │    │                 │    │              │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────┘

Enhancement Logic:
1. Automatic Detection
   ├── Keywords: "summarize", "summary", "overview"
   ├── Patterns: "key points", "main points"
   ├── Phrases: "what are the", "give me an overview"
   └── Actions: "compile", "consolidate", "aggregate"

2. Enhanced Retrieval
   ├── Triple top_k (minimum 15, max 25)
   ├── Increased prefetch (2x normal)
   ├── Better coverage across documents
   └── Smart result diversification

3. Result Diversification Algorithm
   ├── Group results by filename
   ├── Take best result from each file first
   ├── Fill remaining slots with highest scores
   └── Maximize unique file representation

4. Enhanced Instructions
   ├── Multi-source synthesis guidelines
   ├── Theme highlighting across documents
   ├── Structured organization requirements
   └── Coverage maximization instructions
```

## 🔧 Component Details

### ColBERT Embedder

```python
# Singleton pattern for memory efficiency
class ColBERTEmbedder:
    _instance = None
    _initialized = False
    
    # Multi-model support
    - Dense Model: sentence-transformers/all-MiniLM-L6-v2
    - Sparse Model: Qdrant/bm25
    - ColBERT Model: colbert-ir/colbertv2.0
    
    # Key Methods:
    - embed_dense_query(text) → 384d vector
    - embed_sparse_query(text) → sparse vector
    - embed_colbert_query(text) → multi-token vectors
    - embed_dense_documents(texts) → batch dense vectors
    - embed_sparse_documents(texts) → batch sparse vectors
    - embed_colbert_documents(texts) → batch multi-vectors
```

### Qdrant Client Wrapper

```python
class QdrantClientWrapper:
    # Collection Configuration
    - Dense vectors: 384d, COSINE distance, HNSW index
    - Sparse vectors: BM25 with IDF modifier
    - ColBERT vectors: 128d, MAX_SIM comparator, HNSW disabled (m=0)
    
    # Key Features:
    - Auto collection creation
    - Filename field indexing
    - Batch upsert with retry logic
    - Hybrid query with prefetch
    - Late interaction reranking
    
    # Safety Features:
    - Max points per upsert: 16 (configurable)
    - Retry logic with exponential backoff
    - Memory allocation monitoring
    - Graceful error handling
```

### Query Service

```python
class QueryService:
    # Function Calling Tools
    - rag_search(question, top_k, selected_files) → results
    - Local routing for greetings/identity
    - Automatic summary detection
    - Smart result diversification
    
    # Processing Modes:
    - Synchronous: query() → QueryResponse
    - Streaming: stream_answer() → Generator[str]
    - Auto-streaming: stream_answer_auto() → Generator[str]
    
    # Optimizations:
    - Singleton embedder/client reuse
    - Adaptive top_k for summaries
    - Result caching (future)
    - Token limit management
```

## 🚀 Performance Optimizations

### 1. Memory Management
- **Adaptive Batching**: Automatically reduces batch size on OOM errors
- **Singleton Patterns**: Reuses embedding models across requests
- **Progressive Loading**: Models loaded on-demand
- **Memory Monitoring**: Tracks allocation and adjusts accordingly

### 2. Search Optimizations
- **Hybrid Retrieval**: Combines multiple embedding types for accuracy
- **Late Interaction**: ColBERT reranking for precision
- **Smart Diversification**: Spreads results across source files
- **Caching Strategy**: Embeds once, query many times

### 3. Scalability Features
- **Modular Architecture**: Independent, swappable components
- **Async Support**: Non-blocking I/O operations
- **Batch Processing**: Efficient bulk operations
- **Horizontal Scaling**: Stateless service design

### 4. Production Ready
- **Comprehensive Logging**: All operations logged with context
- **Error Handling**: Graceful degradation on failures
- **Health Checks**: Built-in monitoring endpoints
- **Configuration**: Environment-based settings

## 🔍 Advanced Features

### Function Calling Integration
- **Intelligent Routing**: LLM decides when to use RAG
- **Tool Definition**: Structured rag_search function
- **Context Awareness**: Maintains conversation state
- **Fallback Handling**: Direct responses for non-factual queries

### Summary Enhancement
- **Automatic Detection**: Pattern-based summary identification
- **Enhanced Retrieval**: 3x normal result count
- **Smart Diversification**: Cross-file result spreading
- **Comprehensive Coverage**: Multi-source synthesis

### Multi-Modal Support
- **CSV Processing**: Dynamic schema, any column structure
- **PDF Processing**: ColPali visual + text extraction
- **Metadata Preservation**: Full context retention
- **Format Agnostic**: Extensible to new formats

## 📈 Metrics & Monitoring

### Retrieval Metrics
- Number of results retrieved
- Unique files represented
- Search latency
- Reranking effectiveness

### Generation Metrics
- Answer quality scores
- Source attribution accuracy
- Response latency
- Token usage

### System Metrics
- Memory usage
- Processing throughput
- Error rates
- Uptime statistics

---

This architecture supports enterprise-scale document intelligence with the flexibility to adapt to various use cases while maintaining high performance and reliability.