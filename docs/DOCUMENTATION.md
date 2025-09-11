## Qdrant RAG Pipeline - Technical Documentation

### Executive Summary

This project implements an enterprise-grade Retrieval-Augmented Generation (RAG) pipeline featuring advanced hybrid search, intelligent query routing, and production optimizations. Built with FastAPI and Qdrant, it provides state-of-the-art document intelligence capabilities with automatic summary enhancement and smart result diversification.

### Core Innovation

**Advanced Multi-Vector Architecture:**
- **Hybrid Search**: Combines dense semantic vectors, sparse keyword matching, and ColBERT late-interaction reranking
- **Summary Enhancement**: Automatic detection and 3x result expansion for comprehensive queries
- **Smart Diversification**: Cross-file result distribution for better coverage
- **Function Calling**: Intelligent routing between RAG and direct responses

**Production-Ready Features:**
- Memory-efficient adaptive batching to prevent OOM errors
- Singleton pattern for embedding model reuse
- Comprehensive error handling and logging
- Real-time streaming with multiple endpoint options

### Enhanced Hybrid RAG Pipeline: Technical Deep Dive

**1. Advanced Document Ingestion**

   - **Multi-Format Support**: CSV with dynamic schema detection + PDF with ColPali visual processing
   - **Intelligent Chunking**: Token-based segmentation (180 tokens) with configurable overlap and hard limits (512 max)
   - **Dynamic Segmentation**: Content-aware chunking that adapts to document structure (optional)
   - **Memory-Safe Processing**: Adaptive batching that automatically reduces batch size on OOM errors
   - **Three-Vector Generation**: For each chunk, generates:
     - Dense vectors (384d, semantic similarity via MiniLM)
     - Sparse vectors (BM25/keyword matching via Qdrant)
     - ColBERT multivectors (128d per token, late interaction)
   - **Metadata Preservation**: Full payload retention including filename indexing for efficient filtering
   - **Batch Upsert**: Optimized bulk operations with retry logic and exponential backoff

**2. Intelligent Query Processing**

   - **Function Calling Integration**: LLM-based routing that decides when RAG retrieval is needed
   - **Local Routing**: Fast-path handling for greetings and identity queries (if enabled)
   - **Summary Detection**: Automatic identification of comprehensive queries using pattern matching
   - **Query Embedding**: Same three-vector approach as documents for optimal matching
   - **Adaptive Parameters**: Dynamic top_k adjustment based on query type and intent
**3. Advanced Hybrid Retrieval & Reranking**

   - **Prefetch Stage**: Qdrant performs hybrid retrieval using both dense (semantic) and sparse (BM25) vectors
   - **Initial Filtering**: Fast vector similarity search returns top candidates (typically 2-3x final count)
   - **ColBERT Reranking**: Token-level late interaction using MaxSim comparator for precision
   - **Smart Diversification**: For summary queries, applies intelligent cross-file result distribution
   - **Score Fusion**: Combines multiple vector similarity scores into unified ranking
   - **Metadata Filtering**: Supports arbitrary payload-based filtering (filename, tags, etc.)
   - **Quality Assurance**: Returns top-k ranked items with complete source attribution

**4. Enhanced Context Augmentation & LLM Generation**

   - **Context Compilation**: Aggregates retrieved content with relevance scores and source metadata
   - **Summary Enhancement**: For detected summary queries, uses specialized system instructions
   - **Prompt Engineering**: Constructs optimized prompts with guidelines for response style and length
   - **Gemini Integration**: Utilizes Google Gemini 2.5 Flash for high-quality answer generation
   - **Streaming Support**: Real-time token generation with multiple endpoint options (/stream, /stream-auto)
   - **Source Attribution**: Maintains provenance tracking for fact verification and citations
   - **Response Formatting**: Structured output with proper markdown, lists, and source references

**5. Production-Ready Response & Monitoring**

   - **Multi-Format Delivery**: Synchronous JSON responses and streaming server-sent events
   - **Performance Metrics**: Comprehensive timing, source count, and quality indicators
   - **Error Handling**: Graceful degradation with detailed error reporting
   - **Health Monitoring**: Built-in endpoints for system status and resource usage
   - **Logging**: Structured logging with request tracing and performance analytics
6. **Ops & Monitoring**

   - All ingestion and query operations are logged.
   - Collection auto-creation is guarded and logged.
   - Errors are captured and reported in logs and API responses.

### Architecture

- **Backend:** Python (FastAPI)
- **Vector DB:** Qdrant (supports dense and ColBERT multivector fields)
- **Embeddings:** fastembed (TextEmbedding + LateInteractionTextEmbedding for dense + ColBERT multivectors)
- **API:** Modular endpoints for ingestion and query

### Key Features Implemented

1. **Qdrant Multivector Setup**`app/utils/qdrant_multivector_setup.py`: Creates a Qdrant collection with both dense and ColBERT (token-level) multivector fields. Dense uses HNSW; ColBERT disables HNSW (m=0) for efficient reranking.
2. **Embeddings (fastembed)**`app/utils/colbert_embedder.py`: Uses `TextEmbedding` (dense) + `LateInteractionTextEmbedding` (ColBERT) for document and query encoding.
3. **Qdrant Client Wrapper**`app/utils/qdrant_client.py`: Upserts multivector points and performs dense prefetch + ColBERT rerank via `query_points`. Includes auto-create guard + logging.
4. **Ingestion Service**`app/services/ingestion_service.py`: CSV parsing, dynamic payload extraction, multivector embedding, batch upsert with logging & exception capture.
5. **Query Service**`app/services/query_service.py`: Dense + multivector query, reranking, and Gemini-based answer synthesis (stream aggregation) with graceful fallback.
6. **LLM Integration (Gemini)**Optional integration using `google-genai` SDK. Streams answer tokens; builds reasoning summary of source docs.
7. **Logging**`app/utils/logging_config.py`: Central logger config; suppresses noisy dependency logs.
8. **API Endpoints**`app/api/ingestion.py` (`/ingest/csv`), `app/api/query.py` (`/query/`).
9. **Models**`app/models/ingestion.py`, `app/models/query.py`: Pydantic validation.
10. **App Entrypoint**
    `app/main.py`: FastAPI factory + router inclusion.

### Folder Structure

- `app/`
  - `api/` — FastAPI routers
  - `models/` — Pydantic models
  - `services/` — Business logic (ingestion, query, LLM synthesis)
  - `utils/` — Qdrant client, embeddings, logging, collection setup

### Next Steps

- Add streaming FastAPI endpoint for incremental answer tokens
- Add evaluation module (faithfulness, similarity, coverage) per PLAN
- Implement hybrid filtering & metadata faceting
- Add pydantic `BaseSettings` configuration layer
- Introduce unit/integration tests & CI
- Cache embeddings and introduce background ingestion jobs

---

## Complete Project Structure

### Backend: rag-python

This is the main backend application built with FastAPI, providing RAG capabilities with PDF and CSV support.

#### Project Structure

```
rag-python/
├── .env.local                          # Environment variables
├── .gitignore                          # Git ignore file
├── .python-version                     # Python version specification
├── main.py                            # Main entry point
├── pyproject.toml                     # Project dependencies and configuration
├── README.md                          # Project readme
├── requirements.txt                   # Python dependencies
├── app/                               # Main application directory
│   ├── main.py                        # FastAPI application factory
│   ├── api/                           # API endpoints
│   │   ├── files.py                   # File management endpoints (list, delete)
│   │   ├── ingestion.py               # Data ingestion endpoints (CSV, PDF)
│   │   └── query.py                   # Query endpoints (RAG, PDF query, streaming)
│   ├── models/                        # Pydantic models
│   │   ├── ingestion.py               # Ingestion request/response models
│   │   └── query.py                   # Query request/response models
│   ├── services/                      # Business logic services
│   │   ├── files_service.py           # File management service
│   │   ├── ingestion_service.py       # CSV ingestion service
│   │   ├── pdf_query_service.py       # PDF query service with ColPali
│   │   ├── pdf_service.py             # PDF ingestion service with ColPali
│   │   └── query_service.py           # Main query service
│   └── utils/                         # Utility modules
│       ├── colbert_embedder.py        # ColBERT embedding utilities
│       ├── logging_config.py          # Logging configuration
│       ├── pdf_utils.py               # PDF processing utilities
│       ├── qdrant_client.py           # Qdrant client wrapper
│       └── qdrant_multivector_setup.py # Qdrant collection setup
├── build/                             # Build artifacts
└── rag_python.egg-info/              # Package metadata
```

#### Key Features

1. **Modular Architecture**: Following OOP principles with separate services, models, and utilities
2. **PDF Support**: ColPali/ColQwen2-based PDF ingestion and retrieval with scalable multivector approach
3. **CSV Support**: Dynamic schema CSV ingestion with hybrid search
4. **Hybrid RAG**: Dense + sparse + ColBERT multivector retrieval and reranking
5. **File Management**: List and delete files from Qdrant collections
6. **CORS Support**: Configured for frontend-backend communication
7. **Production Quality**: Comprehensive error handling, logging, and modular design

#### API Endpoints

- **Ingestion**:
  - `POST /ingestion/csv` - Ingest CSV files
  - `POST /ingestion/pdf` - Ingest PDF files
- **Query**:
  - `POST /query/` - Main RAG query endpoint
  - `POST /query/pdf` - PDF-specific query with ColPali
  - `POST /query/stream` - Streaming query endpoint
- **Files**:
  - `GET /files/list` - List all uploaded files
  - `DELETE /files/delete` - Delete specific files

#### Dependencies

```toml
dependencies = [
   "fastapi",
   "qdrant-client",
   "pandas",
   "qdrant-client[fastembed]",
   "google-genai",
   "colpali_engine>=0.3.1",
   "PyPDF2",
   "torch",
   "datasets",
]
```

### Frontend: ai-sdk-preview-internal-knowledge-base

This is the Next.js frontend application that provides a clean UI for interacting with the RAG backend.

#### Project Structure

```
ai-sdk-preview-internal-knowledge-base/
├── .env.example                       # Environment variables template
├── .env.local                         # Local environment variables
├── .eslintrc.json                     # ESLint configuration
├── .gitignore                         # Git ignore file
├── LICENSE                            # License file
├── next.config.mjs                    # Next.js configuration
├── package.json                       # Node.js dependencies
├── pnpm-lock.yaml                     # Package lock file
├── postcss.config.mjs                 # PostCSS configuration
├── README.md                          # Project readme
├── tailwind.config.ts                 # Tailwind CSS configuration
├── tsconfig.json                      # TypeScript configuration
├── app/                               # Next.js app directory
│   ├── favicon.ico                    # Favicon
│   ├── globals.css                    # Global styles
│   ├── layout.tsx                     # Root layout component
│   ├── uncut-sans.woff2              # Custom font
│   └── (chat)/                        # Chat feature group
│       ├── opengraph-image.png        # Social media image
│       ├── page.tsx                   # Main chat page
│       ├── twitter-image.png          # Twitter card image
│       ├── [id]/                      # Dynamic chat routes
│       │   └── page.tsx               # Individual chat page
│       └── api/                       # API proxy routes
│           ├── chat/                  # Chat API proxy
│           │   └── route.ts           # Chat endpoint proxy
│           └── files/                 # File management API proxy
│               ├── delete/            # File deletion proxy
│               │   └── route.ts
│               ├── list/              # File listing proxy
│               │   └── route.ts
│               └── upload/            # File upload proxy
│                   └── route.ts
├── components/                        # React components
│   ├── chat.tsx                       # Main chat component
│   ├── data.ts                        # Data utilities
│   ├── files.tsx                      # File management component
│   ├── form.tsx                       # Form components
│   ├── icons.tsx                      # Icon components
│   ├── markdown.tsx                   # Markdown rendering
│   ├── message.tsx                    # Message display component
│   ├── navbar.tsx                     # Navigation bar
│   ├── submit-button.tsx              # Submit button component
│   └── use-scroll-to-bottom.ts        # Scroll hook
├── public/                            # Static assets
│   ├── iphone.png
│   ├── tv.png
│   └── watch.png
└── utils/                             # Utility functions
    ├── functions.ts                   # General utilities with API URL handling
    └── pdf.ts                         # PDF utilities
```

#### Key Features

1. **Pure Frontend**: No backend logic, acts as a proxy to the Python backend
2. **File Upload**: Supports PDF and CSV file uploads with proper validation
3. **Chat Interface**: Clean chat UI for RAG interactions
4. **Responsive Design**: Built with Tailwind CSS for mobile-first design
5. **File Management**: Upload, list, and delete files with visual feedback
6. **Environment Configuration**: Uses `NEXT_PUBLIC_API_URL` for backend communication

#### Removed Features

The following features were removed to make this a pure frontend application:
- Authentication (NextAuth.js)
- Database integration (Drizzle ORM)
- Chat history
- OpenAI integration
- Vercel Blob storage
- Backend API routes (replaced with proxies)

#### Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:3001
```

#### API Proxy Routes

All API routes in the frontend act as proxies to the Python backend:

- `POST /api/chat` → `POST {API_URL}/query`
- `POST /api/files/upload` → `POST {API_URL}/ingestion/csv` or `POST {API_URL}/ingestion/pdf`
- `GET /api/files/list` → `GET {API_URL}/files/list`
- `DELETE /api/files/delete` → `DELETE {API_URL}/files/delete`

## Integration Architecture

```
┌─────────────────────────────────────┐
│           Frontend (Next.js)        │
│  ┌─────────────────────────────────┐ │
│  │        UI Components            │ │
│  │  ┌─────────┐  ┌─────────────┐   │ │
│  │  │  Chat   │  │    Files    │   │ │
│  │  └─────────┘  └─────────────┘   │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │       API Proxy Routes          │ │
│  │  ┌─────────┐  ┌─────────────┐   │ │
│  │  │ /chat   │  │   /files    │   │ │
│  │  └─────────┘  └─────────────┘   │ │
│  └─────────────────────────────────┘ │
└─────────────────┬───────────────────┘
                  │ HTTP Requests
                  │ (NEXT_PUBLIC_API_URL)
                  ▼
┌─────────────────────────────────────┐
│         Backend (FastAPI)           │
│  ┌─────────────────────────────────┐ │
│  │           API Routes            │ │
│  │  ┌──────────┐ ┌──────────────┐  │ │
│  │  │ /query   │ │ /ingestion   │  │ │
│  │  │ /files   │ │              │  │ │
│  │  └──────────┘ └──────────────┘  │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │          Services               │ │
│  │  ┌──────────┐ ┌──────────────┐  │ │
│  │  │   RAG    │ │     PDF      │  │ │
│  │  │ Service  │ │   Service    │  │ │
│  │  └──────────┘ └──────────────┘  │ │
│  └─────────────────────────────────┘ │
└─────────────────┬───────────────────┘
                  │
                  ▼
            ┌─────────────┐
            │   Qdrant    │
            │  Vector DB  │
            └─────────────┘
```

## Development Workflow

1. **Start Backend**: Run the FastAPI server from `rag-python/`
2. **Start Frontend**: Run the Next.js development server from `ai-sdk-preview-internal-knowledge-base/`
3. **Configure Environment**: Set `NEXT_PUBLIC_API_URL` to point to the FastAPI server
4. **Upload Files**: Use the UI to upload PDF or CSV files
5. **Query**: Use the chat interface to ask questions about uploaded documents

---

This documentation will be updated as the pipeline evolves.
