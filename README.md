# Evolutyz RAG Python - Advanced Document Intelligence Platform

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, featuring hybrid search, multi-modal embeddings, and intelligent document processing capabilities.

## 🚀 Overview

This RAG application provides state-of-the-art document intelligence through a sophisticated multi-vector search architecture. It combines dense semantic search, sparse keyword matching, and ColBERT late-interaction reranking to deliver highly accurate and contextually relevant answers from your documents.

### Key Highlights

- **Hybrid Search Architecture**: Combines dense, sparse, and ColBERT embeddings for maximum retrieval precision
- **Intelligent Document Processing**: Supports both CSV and PDF files with dynamic schema adaptation
- **Advanced Query Processing**: Automatic summary detection, smart result diversification, and function calling
- **Production-Ready**: Memory-efficient batching, comprehensive logging, and robust error handling
- **Scalable Design**: Modular architecture with clean separation of concerns

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  API Layer     │  /query  │  /ingestion  │  /files         │
│  Services      │  Query   │  Ingestion   │  File Mgmt      │
│  Utils         │  Embeddings │ Qdrant Client │ Logging      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Database                   │
│  Dense Vectors (MiniLM) + Sparse Vectors (BM25) +          │
│  ColBERT Multivectors (Late Interaction)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Google Gemini 2.5 Flash                 │
│              Answer Generation & Synthesis                  │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Core Features

### 1. **Multi-Modal Document Ingestion**
- **CSV Support**: Dynamic schema detection and processing
- **PDF Support**: ColPali/ColQwen2-based visual document understanding
- **Intelligent Chunking**: Token-based segmentation with overlap control
- **Metadata Preservation**: Full payload retention for filtering and context

### 2. **Hybrid Search Engine**
- **Dense Embeddings**: Semantic understanding via sentence-transformers
- **Sparse Embeddings**: Keyword matching through BM25
- **ColBERT Reranking**: Token-level late interaction for precision
- **Multi-Vector Fusion**: Optimized ranking combining all embedding types

### 3. **Advanced Query Processing**
- **Automatic Summary Detection**: Enhanced retrieval for comprehensive overviews
- **Smart Diversification**: Result spreading across multiple source files
- **Function Calling**: Intelligent routing between RAG and direct responses
- **Streaming Support**: Real-time answer generation with multiple endpoints

### 4. **Production Optimizations**
- **Memory Management**: Adaptive batching to prevent OOM errors
- **Performance Caching**: Singleton patterns for embedding models
- **Robust Error Handling**: Comprehensive logging and graceful degradation
- **Scalable Architecture**: Modular design supporting horizontal scaling

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Qdrant database (local or cloud)
- Google Gemini API key

### 1. Clone the Repository
```bash
git clone https://github.com/aksisonline/evolutyz-rag-python.git
cd evolutyz-rag-python
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or using the modern approach:
pip install -e .
```

### 3. Environment Configuration
Copy `.env.example` to `.env.local` and configure:

```bash
# Core Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
COLLECTION_NAME=rag_collection

# Embedding Models
DENSE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COLBERT_EMBEDDING_MODEL=colbert-ir/colbertv2.0
SPARSE_EMBEDDING_MODEL=Qdrant/bm25

# Performance Tuning
INGEST_BATCH_SIZE=64
CSV_CHUNK_TOKENS=180
CSV_CHUNK_MAX_TOKENS=300
```

### 4. Start the Application
```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📖 Usage Guide

### Document Ingestion

#### CSV Files
```bash
curl -X POST "http://localhost:8000/ingestion/csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.csv"
```

#### PDF Files
```bash
curl -X POST "http://localhost:8000/ingestion/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### Querying Documents

#### Standard Query
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main features of this product?",
    "top_k": 5,
    "style": "detailed"
  }'
```

#### Streaming Query
```bash
curl -X POST "http://localhost:8000/query/stream-auto" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key points from all documents",
    "top_k": 15,
    "use_function_calling": true
  }'
```

### File Management
```bash
# List all files
curl -X GET "http://localhost:8000/files/list"

# Delete specific files
curl -X DELETE "http://localhost:8000/files/delete" \
  -H "Content-Type: application/json" \
  -d '{"filenames": ["document1.pdf", "data.csv"]}'
```

## 🧠 How It Works

### 1. **Document Processing Pipeline**
1. Upload documents via API endpoints
2. Extract and preprocess text content
3. Generate three types of embeddings:
   - Dense vectors for semantic similarity
   - Sparse vectors for keyword matching
   - ColBERT multivectors for token-level precision
4. Store in Qdrant with metadata preservation

### 2. **Query Processing Pipeline**
1. Receive user question
2. Detect query type (summary, specific question, etc.)
3. Generate query embeddings using the same models
4. Perform hybrid retrieval from Qdrant
5. Apply ColBERT reranking for precision
6. Diversify results for better coverage (if needed)
7. Generate contextual answer using Gemini

### 3. **Advanced Features**

#### Summary Enhancement
The system automatically detects summary requests using keywords like "summarize", "overview", "main points", etc., and:
- Increases retrieval count (3x normal)
- Applies smart diversification across files
- Uses enhanced system instructions for comprehensive coverage

#### Smart Diversification
For comprehensive queries, the system:
- Takes the best result from each source file first
- Fills remaining slots with highest-scoring results
- Maximizes unique file representation in results

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARY_MAX_TOP_K` | 25 | Maximum results for summary requests |
| `INGEST_BATCH_SIZE` | 64 | Batch size for document processing |
| `CSV_CHUNK_TOKENS` | 180 | Target tokens per chunk |
| `QDRANT_MAX_POINTS_PER_UPSERT` | 16 | Max points per Qdrant upsert |
| `STYLE_DETAILED_MAX_WORDS` | 180 | Word limit for detailed responses |

### Query Styles
- **minimal**: Concise answers (≤25 words)
- **concise**: Brief responses (≤60 words)  
- **detailed**: Comprehensive answers (≤180 words)

## 🚀 API Reference

### Ingestion Endpoints

#### `POST /ingestion/csv`
Upload and process CSV files
- **Input**: Multipart form with CSV file
- **Output**: Success confirmation with processing stats

#### `POST /ingestion/pdf`
Upload and process PDF files
- **Input**: Multipart form with PDF file
- **Output**: Success confirmation with processing stats

### Query Endpoints

#### `POST /query/`
Synchronous query processing
- **Input**: `QueryRequest` with question, top_k, style, etc.
- **Output**: `QueryResponse` with answer, sources, and metrics

#### `POST /query/stream`
Legacy streaming endpoint
- **Input**: `QueryRequest`
- **Output**: Server-sent events with streaming response

#### `POST /query/stream-auto`
Advanced streaming with function calling
- **Input**: `QueryRequest` + `use_function_calling` flag
- **Output**: Enhanced streaming with intelligent routing

### File Management Endpoints

#### `GET /files/list`
List all uploaded files
- **Output**: Array of file information with counts and metadata

#### `DELETE /files/delete`
Delete specific files from the collection
- **Input**: Array of filenames to delete
- **Output**: Deletion confirmation

## 🧪 Testing

### Run Integration Tests
```bash
# Test function calling integration
python test_function_calling_integration.py

# Test summary enhancement features
python test_summary_enhancement.py
```

### Manual Testing
1. Start the application
2. Upload test documents via `/ingestion/csv` or `/ingestion/pdf`
3. Query using different styles and parameters
4. Verify streaming endpoints work correctly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the existing documentation in `/docs/`
2. Review the configuration options
3. Check logs for detailed error information
4. Open an issue on GitHub with detailed reproduction steps

---

*Built with ❤️ by the Evolutyz team for intelligent document processing*