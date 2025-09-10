## Qdrant RAG Pipeline Documentation

### Overview

This project implements a modular, production-quality Retrieval-Augmented Generation (RAG) pipeline using FastAPI, Qdrant, and ColBERT-style multivector reranking (dense + late interaction). The codebase is organized for scalability and maintainability, following OOP and best practices.

**Hybrid search and reranking is fully implemented:**

- The pipeline uses Qdrant's `query_points` with `prefetch` for both dense (semantic) and sparse (BM25/keyword) retrieval, followed by ColBERT (late-interaction) reranking for every query.
- ColBERT token-level vectors are used for reranking the hybrid-retrieved candidates, ensuring high-precision semantic matching.

### Hybrid RAG Pipeline: Step-by-Step Flow

1. **Data Ingestion**

   - User uploads a CSV via the API.
   - Backend parses each row, dynamically creates a payload from all columns (schema-flexible).
   - Text columns are concatenated for embedding.
   - For each row, three types of embeddings are generated:
     - Dense (semantic, e.g., MiniLM)
     - Sparse (BM25/keyword)
     - ColBERT (token-level, late interaction)
   - Qdrant upserts: each point contains all three vector types and full payload metadata (batch operation).
2. **Query Processing**

   - User submits a question via the API.
   - Backend encodes the query as dense, sparse, and ColBERT embeddings.
3. **Hybrid Retrieval & Reranking**

   - Qdrant performs hybrid retrieval using both dense (semantic) and sparse (BM25) vectors via `prefetch`.
   - The top candidates are reranked using ColBERT token-level late interaction (MaxSim comparator).
   - Supports arbitrary metadata filtering via payload.
   - Returns top-k ranked items with all payload metadata.
4. **Context Augmentation & LLM Generation**

   - Backend compiles retrieved contexts and aggregates relevant payload fields.
   - Builds an augmented prompt for Gemini 2.5 Flash (if enabled) with guidelines for concise responses (150-200 words max).
   - LLM generates concise, well-structured answers with proper formatting, reasoning, and source citations (streamed if supported).
5. **Response**

   - API returns the answer, reasoning, and sources to the frontend.
   - (Optional) Streaming endpoint can be used for real-time answer delivery.
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
