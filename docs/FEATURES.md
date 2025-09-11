# Complete Features & Functions Guide

## 🌟 Feature Overview

The Evolutyz RAG Python application provides a comprehensive suite of features for intelligent document processing and retrieval-augmented generation. This guide details every feature, function, and capability of the system.

## 📋 Table of Contents
1. [Document Ingestion Features](#document-ingestion-features)
2. [Search & Retrieval Features](#search--retrieval-features)
3. [Query Processing Features](#query-processing-features)
4. [Advanced AI Features](#advanced-ai-features)
5. [API Features](#api-features)
6. [Configuration Features](#configuration-features)
7. [Performance Features](#performance-features)
8. [Monitoring Features](#monitoring-features)

---

## 📄 Document Ingestion Features

### 1. **Multi-Format Document Support**

#### CSV File Processing
- **Dynamic Schema Detection**: Automatically adapts to any CSV structure
- **Column Mapping**: All columns preserved as metadata
- **Text Concatenation**: Intelligent combination of text columns for embedding
- **Data Type Inference**: Automatic detection of data types per column
- **Large File Handling**: Memory-efficient streaming for big datasets

**Supported Features:**
```python
# Environment Configuration
CSV_CHUNK_TOKENS=180          # Target tokens per chunk
CSV_CHUNK_MAX_TOKENS=300      # Maximum tokens per chunk
CSV_CHUNK_OVERLAP_TOKENS=0    # Overlap between chunks
CSV_CHUNK_HARD_MAX_TOKENS=512 # Hard limit per chunk
CSV_DYNAMIC_SEGMENT=1         # Enable dynamic segmentation
```

#### PDF File Processing
- **Visual Understanding**: ColPali/ColQwen2-based visual processing
- **Text Extraction**: Multi-layer text extraction
- **Page Preservation**: Page number tracking and metadata
- **Image Processing**: Visual elements understanding
- **Layout Awareness**: Structure-preserving processing

**Supported Features:**
```python
# PDF-specific Configuration
PDF_CHUNK_TOKENS=180          # Target tokens per chunk
PDF_CHUNK_MAX_TOKENS=300      # Maximum tokens per chunk
PDF_BATCH_SIZE=64             # Batch size for processing
```

### 2. **Intelligent Text Chunking**

#### Static Token-Based Chunking
- **Configurable Size**: Adjustable token limits per chunk
- **Overlap Control**: Configurable overlap between chunks
- **Hard Limits**: Maximum token enforcement
- **Context Preservation**: Maintains semantic boundaries

#### Dynamic Segmentation
- **Adaptive Chunking**: Content-aware segmentation
- **Target Segments**: Configurable number of segments per document
- **Minimum/Maximum Tokens**: Flexible size constraints
- **Content-Aware Splitting**: Semantic boundary detection

**Configuration Options:**
```python
# Dynamic Segmentation
CSV_DYNAMIC_TARGET_SEGMENTS=12
CSV_DYNAMIC_MIN_TOKENS=120
CSV_DYNAMIC_MAX_TOKENS=300
```

### 3. **Memory-Efficient Processing**

#### Adaptive Batching
- **OOM Protection**: Automatic batch size reduction on memory errors
- **Progressive Scaling**: Starts with large batches, scales down if needed
- **Minimum Batch Size**: Configurable minimum (default: 1)
- **Recovery Logic**: Automatic retry with smaller batches

#### Streaming Processing
- **Large File Support**: Processes files larger than available memory
- **Chunked Reading**: Pandas chunksize-based processing
- **Progress Tracking**: Real-time processing status
- **Error Recovery**: Graceful handling of partial failures

---

## 🔍 Search & Retrieval Features

### 1. **Hybrid Search Architecture**

#### Multi-Vector Search
- **Dense Vectors**: Semantic similarity using sentence-transformers
- **Sparse Vectors**: Keyword matching using BM25
- **ColBERT Vectors**: Token-level late interaction
- **Score Fusion**: Intelligent combination of all vector types

**Vector Specifications:**
```python
Dense Vectors:     384 dimensions (all-MiniLM-L6-v2)
Sparse Vectors:    BM25 with IDF modifier
ColBERT Vectors:   128 dimensions per token (colbertv2.0)
```

#### Advanced Reranking
- **Late Interaction**: ColBERT token-level similarity
- **MaxSim Comparator**: Optimal token matching
- **Hybrid Prefetch**: Initial filtering with dense+sparse
- **Final Reranking**: ColBERT precision refinement

### 2. **Smart Result Processing**

#### Result Diversification
- **File Distribution**: Spreads results across multiple source files
- **Quality Preservation**: Maintains relevance while diversifying
- **Coverage Optimization**: Maximizes information breadth
- **Configurable Limits**: Adjustable diversification parameters

**Algorithm Details:**
```python
def _diversify_results_by_file(results, target_k):
    # 1. Group results by filename
    # 2. Take best result from each file first
    # 3. Fill remaining slots with highest scores
    # 4. Maximize unique file representation
```

#### Filtering & Selection
- **File-Based Filtering**: Query specific documents
- **Metadata Filtering**: Filter by any document property
- **Score Thresholding**: Quality-based result filtering
- **Top-K Selection**: Configurable result count

### 3. **Query-Adaptive Retrieval**

#### Summary Enhancement
- **Automatic Detection**: Identifies summary requests
- **Enhanced Retrieval**: 3x normal result count
- **Cross-File Coverage**: Ensures diverse source representation
- **Comprehensive Context**: Maximum information synthesis

**Detection Keywords:**
```python
Summary Keywords: ["summarize", "summary", "summarise"]
Overview Keywords: ["overview", "key points", "main points"]
Question Keywords: ["what are the", "what do the documents"]
Action Keywords: ["compile", "consolidate", "aggregate"]
```

#### Context-Aware Processing
- **Query Intent Detection**: Understands query purpose
- **Adaptive Parameters**: Adjusts retrieval based on intent
- **Style-Aware Processing**: Matches response style to query
- **Dynamic Top-K**: Intelligent result count adjustment

---

## 🤖 Query Processing Features

### 1. **Function Calling Integration**

#### Intelligent Routing
- **LLM-Based Decisions**: Gemini determines when to use RAG
- **Local Routing**: Fast handling of greetings/identity queries
- **Fallback Handling**: Direct responses for non-factual queries
- **Context Preservation**: Maintains conversation state

**Routing Logic:**
```python
# Local routing (if enabled)
if is_greeting_or_identity(query):
    return direct_response(query)

# LLM-based routing (default)
if llm_needs_information(query):
    return rag_search(query) + llm_response(context)
else:
    return direct_llm_response(query)
```

#### Tool Definition
- **Structured Functions**: Well-defined RAG search tool
- **Parameter Validation**: Type checking and constraints
- **Error Handling**: Graceful tool failure recovery
- **Performance Monitoring**: Tool usage metrics

### 2. **Multi-Modal Query Processing**

#### Query Types
- **Factual Questions**: Information retrieval from documents
- **Summary Requests**: Comprehensive document overviews
- **Analytical Queries**: Cross-document analysis
- **Comparative Questions**: Multi-source comparisons

#### Response Styles
- **Minimal**: Concise answers (≤25 words)
- **Concise**: Brief responses (≤60 words)
- **Detailed**: Comprehensive answers (≤180 words)
- **Custom**: User-defined word limits

### 3. **Streaming Capabilities**

#### Multiple Streaming Endpoints
- **Legacy Streaming** (`/stream`): Basic token streaming
- **Auto Streaming** (`/stream-auto`): Function calling with streaming
- **Synchronous** (`/query`): Traditional request-response

#### Real-Time Processing
- **Token-by-Token**: Live response generation
- **Progress Indicators**: Processing status updates
- **Cancellation Support**: Abort long-running queries
- **Buffer Management**: Efficient memory usage

---

## 🧠 Advanced AI Features

### 1. **Google Gemini Integration**

#### Model Configuration
- **Gemini 2.5 Flash**: High-performance language model
- **Streaming Support**: Real-time token generation
- **Function Calling**: Tool integration capabilities
- **Context Management**: Large context window utilization

#### Prompt Engineering
- **System Instructions**: Task-specific guidance
- **Context Formatting**: Optimal information presentation
- **Source Attribution**: Citation requirements
- **Style Enforcement**: Response format control

### 2. **Embedding Models**

#### Dense Embeddings
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Use Case**: Semantic similarity
- **Performance**: Fast, accurate semantic matching

#### Sparse Embeddings
- **Model**: Qdrant/bm25
- **Type**: Traditional keyword matching
- **Use Case**: Exact term matching
- **Performance**: Highly efficient for specific terms

#### ColBERT Embeddings
- **Model**: colbert-ir/colbertv2.0
- **Type**: Late interaction multi-vector
- **Dimensions**: 128 per token
- **Use Case**: Precision reranking

### 3. **AI Optimizations**

#### Caching Strategy
- **Model Singletons**: Reuse loaded models
- **Embedding Cache**: Cache computed embeddings
- **Query Cache**: Store frequent query results
- **Performance Boost**: Significant latency reduction

#### Memory Management
- **Lazy Loading**: Load models on demand
- **Memory Monitoring**: Track usage patterns
- **Garbage Collection**: Automatic cleanup
- **Resource Optimization**: Efficient GPU/CPU usage

---

## 🚀 API Features

### 1. **RESTful API Design**

#### Endpoint Categories
- **Ingestion Endpoints** (`/ingestion/*`): Document upload and processing
- **Query Endpoints** (`/query/*`): Search and question answering
- **File Management** (`/files/*`): Document lifecycle management
- **Health Endpoints**: System status and monitoring

#### Request/Response Formats
- **JSON Payloads**: Structured data exchange
- **Multipart Forms**: File upload support
- **Server-Sent Events**: Streaming responses
- **Error Responses**: Detailed error information

### 2. **Authentication & Security**

#### CORS Configuration
- **Cross-Origin Support**: Frontend integration
- **Configurable Origins**: Environment-based control
- **Header Management**: Custom header support
- **Credential Handling**: Secure authentication

#### Input Validation
- **Pydantic Models**: Type-safe request validation
- **File Type Checking**: MIME type validation
- **Size Limits**: File size constraints
- **Content Validation**: Malicious content detection

### 3. **API Documentation**

#### OpenAPI/Swagger
- **Interactive Documentation**: Auto-generated API docs
- **Schema Definitions**: Complete model documentation
- **Example Requests**: Sample API calls
- **Response Schemas**: Expected output formats

#### Usage Examples
- **curl Commands**: Command-line examples
- **Python Clients**: SDK-style usage
- **JavaScript**: Frontend integration examples
- **Postman Collections**: Ready-to-use API tests

---

## ⚙️ Configuration Features

### 1. **Environment-Based Configuration**

#### Core Settings
```python
# Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API=your_api_key
COLLECTION_NAME=rag_collection

# AI Model Configuration
GEMINI_API_KEY=your_gemini_key
DENSE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COLBERT_EMBEDDING_MODEL=colbert-ir/colbertv2.0
SPARSE_EMBEDDING_MODEL=Qdrant/bm25
```

#### Performance Tuning
```python
# Batch Processing
INGEST_BATCH_SIZE=64
PDF_BATCH_SIZE=64
QDRANT_MAX_POINTS_PER_UPSERT=16

# Memory Management
QDRANT_UPSERT_RETRIES=3
QDRANT_UPSERT_BACKOFF_BASE=0.5
```

### 2. **Feature Toggles**

#### Optional Features
```python
# Enable/Disable Features
DISABLE_FUNCTION_ROUTING_FOR_QUERY=1
ENABLE_LOCAL_ROUTING=1
ENABLE_EAGER_RETRIEVAL=1
CSV_DYNAMIC_SEGMENT=1
PDF_DYNAMIC_SEGMENT=1
QDRANT_DISABLE_COLBERT=1
```

#### Style Configuration
```python
# Response Word Limits
STYLE_MINIMAL_MAX_WORDS=25
STYLE_CONCISE_MAX_WORDS=60
STYLE_DETAILED_MAX_WORDS=180
```

### 3. **Advanced Configuration**

#### Summary Enhancement
```python
SUMMARY_MAX_TOP_K=25
METRICS_MAX_FILE_LIST=12
```

#### Text Processing
```python
# Storage Options
STORE_CHUNK_TEXT=1
CHUNK_TEXT_FIELD=text
CHUNK_TEXT_MAX_CHARS=1600
FULL_CHUNK_TEXT_FIELD=text_full
FULL_CHUNK_TEXT_MAX_CHARS=8000
```

---

## 🚀 Performance Features

### 1. **Scalability**

#### Horizontal Scaling
- **Stateless Design**: No server-side state management
- **Load Balancer Ready**: Multiple instance support
- **Database Sharding**: Qdrant collection distribution
- **Microservice Architecture**: Component independence

#### Vertical Scaling
- **Resource Optimization**: Efficient CPU/memory usage
- **GPU Support**: Accelerated embedding computation
- **Parallel Processing**: Multi-threaded operations
- **Cache Utilization**: Memory and disk caching

### 2. **Optimization Techniques**

#### Batch Processing
- **Vectorized Operations**: NumPy/PyTorch optimizations
- **Bulk Database Operations**: Reduced I/O overhead
- **Pipeline Processing**: Overlapped computation stages
- **Memory Pooling**: Efficient memory allocation

#### Caching Strategies
- **Model Caching**: Persistent model loading
- **Result Caching**: Query result storage
- **Embedding Caching**: Precomputed vectors
- **Configuration Caching**: Runtime optimization

### 3. **Resource Management**

#### Memory Optimization
- **Adaptive Batching**: Dynamic size adjustment
- **Garbage Collection**: Automatic cleanup
- **Memory Monitoring**: Usage tracking
- **Leak Prevention**: Resource lifecycle management

#### CPU/GPU Utilization
- **Multi-threading**: Parallel execution
- **GPU Acceleration**: CUDA support
- **Load Balancing**: Resource distribution
- **Priority Queuing**: Task scheduling

---

## 📊 Monitoring Features

### 1. **Logging & Observability**

#### Comprehensive Logging
- **Structured Logs**: JSON-formatted log entries
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Context Preservation**: Request ID tracking
- **Performance Metrics**: Latency and throughput

#### Monitoring Points
- **Request Processing**: End-to-end timing
- **Database Operations**: Query performance
- **AI Model Calls**: Token usage and latency
- **Error Tracking**: Failure analysis

### 2. **Metrics Collection**

#### System Metrics
- **Memory Usage**: Current and peak usage
- **CPU Utilization**: Processing load
- **Network I/O**: Request/response sizes
- **Disk Usage**: Storage consumption

#### Business Metrics
- **Query Volume**: Request counts
- **Document Processing**: Ingestion statistics
- **User Patterns**: Usage analytics
- **Quality Metrics**: Answer satisfaction

### 3. **Health Checks**

#### Service Health
- **Database Connectivity**: Qdrant status
- **Model Availability**: AI service health
- **API Responsiveness**: Endpoint monitoring
- **Resource Availability**: System capacity

#### Alerting
- **Error Rate Monitoring**: Failure thresholds
- **Performance Degradation**: Latency alerts
- **Resource Exhaustion**: Capacity warnings
- **Service Dependencies**: External service status

---

## 🔮 Future Features (Roadmap)

### 1. **Enhanced AI Capabilities**
- **Multi-Modal Understanding**: Image + text processing
- **Conversation Memory**: Chat history integration
- **Advanced RAG**: Graph-based retrieval
- **Custom Models**: Fine-tuned embeddings

### 2. **Enterprise Features**
- **Authentication**: OAuth2/SAML integration
- **Multi-Tenancy**: Organization isolation
- **Audit Logging**: Compliance tracking
- **Data Governance**: Access control

### 3. **Performance Improvements**
- **Vector Caching**: Persistent embedding storage
- **Query Optimization**: Adaptive indexing
- **Distributed Processing**: Multi-node deployment
- **Real-Time Updates**: Incremental indexing

---

This comprehensive feature guide provides complete coverage of all capabilities in the Evolutyz RAG Python application. Each feature is designed for production use with enterprise-grade reliability and performance.