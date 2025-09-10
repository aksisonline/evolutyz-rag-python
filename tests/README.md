# RAG System Tests

Comprehensive tests for the RAG (Retrieval-Augmented Generation) system core services.

## Quick Start

### Prerequisites
- Python virtual environment activated: `.venv/Scripts/activate`
- Dependencies installed: `pip install -r requirements.txt`
- Qdrant running on `http://localhost:6333` ✅

### Run All Tests
```bash
python -m pytest tests/ -v
```

## Test Commands

### Core Service Tests (No External Services)
```bash
# Essential service functionality tests
python -m pytest tests/test_markdown_processing.py tests/test_file_selection_logic.py tests/test_empty_file_selection.py -v
```

### Integration Tests (Requires Qdrant)
```bash
# Database connectivity tests
python -m pytest tests/test_qdrant_connection.py -v

# Complete service integration tests
python -m pytest tests/test_integration_check.py tests/test_delete_functionality.py -v
```

### Service-Specific Tests
```bash
# QueryService tests
python -m pytest tests/test_empty_file_selection.py tests/test_file_selection_logic.py -v

# TextProcessingService and LLMService tests
python -m pytest tests/test_markdown_processing.py tests/test_integration_check.py -v

# FilesService tests
python -m pytest tests/test_delete_functionality.py -v
```

## Core Service Tests Overview

- `test_markdown_processing.py` - **TextProcessingService**, **LLMService**, **QueryService** comprehensive tests
- `test_integration_check.py` - Service integration and component interaction tests
- `test_empty_file_selection.py` - **QueryService** empty file selection and RAG bypass logic
- `test_file_selection_logic.py` - **QueryService** file filtering and selection logic
- `test_qdrant_connection.py` - Vector database connectivity and infrastructure
- `test_delete_functionality.py` - **FilesService** document deletion functionality

## Services Tested

### QueryService

The main orchestration service that handles:

- File selection filtering and validation
- RAG operations coordination
- Empty file selection bypass logic
- Integration with embedding and vector search

### TextProcessingService

Text processing and analysis service:

- Markdown and text cleaning
- Context building and structuring
- Text similarity calculations
- Keyword extraction

### LLMService

Large Language Model integration service:

- Response synthesis and generation
- Prompt building and optimization
- Greeting and error responses
- Stream processing

### FilesService

File management service:

- Document deletion and cleanup
- File metadata handling
- Storage operations

## Environment Configuration (Optional)

For LLM tests, add to `.env.local`:

```bash
GOOGLE_API_KEY=your_api_key_here
```

## Coverage Report

```bash
python -m pytest tests/ --cov=app --cov-report=html
```

## Common Issues

- **Import errors**: Run from project root with virtual environment activated
- **Qdrant connection**: Ensure Qdrant is running on port 6333
- **Missing API key**: LLM tests will skip if no Google API key provided

## Test Statistics

- **Total Tests**: 23 core service tests
- **Services Covered**: QueryService, TextProcessingService, LLMService, FilesService
- **Infrastructure**: Qdrant vector database connectivity
