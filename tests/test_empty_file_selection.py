#!/usr/bin/env python3
"""
Test that the system handles empty file selection correctly
without performing unnecessary RAG operations.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.services.query_service import QueryService
from app.models.query import QueryRequest


def test_empty_file_selection_no_rag():
    """Test that no RAG operations are performed when no files are selected"""
    
    # Mock the dependencies
    with patch('app.services.query_service.QdrantClientWrapper') as mock_qdrant, \
         patch('app.services.query_service.ColBERTEmbedder') as mock_embedder, \
         patch('app.services.query_service.TextProcessingService') as mock_text_processor, \
         patch('app.services.query_service.LLMService') as mock_llm:
        
        query_service = QueryService()
        
        # Test with empty selected_files list
        request = QueryRequest(
            question="What is the summary of these documents?",
            filters={"selected_files": []},  # Empty list
            top_k=5
        )
        
        response = query_service.query(request)
        
        # Verify response indicates no files selected
        assert "no files are currently selected" in response.answer.lower()
        assert response.sources == []
        assert response.reasoning == "No files selected"
        
        # Verify that no embedding or vector operations were called
        mock_embedder.return_value.embed_dense_query.assert_not_called()
        mock_embedder.return_value.embed_sparse_query.assert_not_called()
        mock_embedder.return_value.embed_colbert_query.assert_not_called()
        mock_qdrant.return_value.query_hybrid_with_rerank.assert_not_called()
        mock_text_processor.return_value.build_structured_context.assert_not_called()
        mock_llm.return_value.synthesize_answer.assert_not_called()


def test_empty_file_selection_streaming():
    """Test that streaming also handles empty file selection correctly"""
    
    with patch('app.services.query_service.QdrantClientWrapper') as mock_qdrant, \
         patch('app.services.query_service.ColBERTEmbedder') as mock_embedder, \
         patch('app.services.query_service.TextProcessingService') as mock_text_processor, \
         patch('app.services.query_service.LLMService') as mock_llm:
        
        query_service = QueryService()
        
        # Test with empty selected_files list
        request = QueryRequest(
            question="What is the summary of these documents?",
            filters={"selected_files": []},  # Empty list
            top_k=5
        )
        
        # Collect streaming response
        response_chunks = list(query_service.stream_answer(request))
        full_response = "".join(response_chunks)
        
        # Verify response indicates no files selected
        assert "no files are currently selected" in full_response.lower()
        
        # Verify that no expensive operations were called
        mock_embedder.return_value.embed_dense_query.assert_not_called()
        mock_embedder.return_value.embed_sparse_query.assert_not_called()
        mock_embedder.return_value.embed_colbert_query.assert_not_called()
        mock_qdrant.return_value.query_hybrid_with_rerank.assert_not_called()
        mock_text_processor.return_value.build_structured_context.assert_not_called()
        mock_llm.return_value.stream_answer_with_metrics.assert_not_called()


def test_greeting_with_empty_files():
    """Test that greetings are handled appropriately when no files are selected"""
    
    with patch('app.services.query_service.QdrantClientWrapper'), \
         patch('app.services.query_service.ColBERTEmbedder'), \
         patch('app.services.query_service.TextProcessingService'), \
         patch('app.services.query_service.LLMService'):
        
        query_service = QueryService()
        
        # Test greeting with empty files
        request = QueryRequest(
            question="hi",
            filters={"selected_files": []},
            top_k=5
        )
        
        response = query_service.query(request)
        
        # Should get a greeting response mentioning file selection
        assert "Hello! I'm IRA" in response.answer
        assert "select one or more files" in response.answer
        assert response.reasoning == "No files selected"


def test_normal_query_with_files_selected():
    """Test that normal RAG operations occur when files are selected"""
    
    with patch('app.services.query_service.QdrantClientWrapper') as mock_qdrant, \
         patch('app.services.query_service.ColBERTEmbedder') as mock_embedder, \
         patch('app.services.query_service.TextProcessingService') as mock_text_processor, \
         patch('app.services.query_service.LLMService') as mock_llm:
        
        # Mock the returns
        mock_embedder.return_value.embed_dense_query.return_value = [0.1, 0.2, 0.3]
        mock_embedder.return_value.embed_sparse_query.return_value = {"sparse": "vector"}
        mock_embedder.return_value.embed_colbert_query.return_value = "colbert_query"
        
        mock_result = MagicMock()
        mock_result.payload = {"text": "test content", "filename": "test.txt"}
        mock_result.score = 0.95
        mock_qdrant.return_value.query_hybrid_with_rerank.return_value = [mock_result]
        
        mock_text_processor.return_value.build_structured_context.return_value = "test context"
        mock_text_processor.return_value.calculate_text_similarity.return_value = 0.8
        mock_llm.return_value.synthesize_answer.return_value = ("test answer", "test reasoning")
        
        query_service = QueryService()
        
        # Test with files selected
        request = QueryRequest(
            question="What is the summary?",
            filters={"selected_files": ["document1.pdf"]},  # Files selected
            top_k=5
        )
        
        response = query_service.query(request)
        
        # Verify RAG operations were called
        mock_embedder.return_value.embed_dense_query.assert_called_once()
        mock_embedder.return_value.embed_sparse_query.assert_called_once()
        mock_embedder.return_value.embed_colbert_query.assert_called_once()
        mock_qdrant.return_value.query_hybrid_with_rerank.assert_called_once()
        mock_text_processor.return_value.build_structured_context.assert_called_once()
        mock_llm.return_value.synthesize_answer.assert_called_once()
        
        assert response.answer == "test answer"
        assert response.reasoning == "test reasoning"
        assert len(response.sources) == 1


def test_empty_filters_no_rag():
    """Test that no RAG operations are performed when filters are empty dict"""
    
    # Mock the dependencies
    with patch('app.services.query_service.QdrantClientWrapper') as mock_qdrant, \
         patch('app.services.query_service.ColBERTEmbedder') as mock_embedder, \
         patch('app.services.query_service.TextProcessingService') as mock_text_processor, \
         patch('app.services.query_service.LLMService') as mock_llm:
        
        query_service = QueryService()
        
        # Test with empty filters dict (as frontend sends)
        request = QueryRequest(
            question="What is the summary of these documents?",
            filters={},  # Empty dict instead of {"selected_files": []}
            top_k=5
        )
        
        response = query_service.query(request)
        
        # Verify response indicates no files selected
        assert "no files are currently selected" in response.answer.lower()
        assert response.sources == []
        assert response.reasoning == "No files selected"
        
        # Verify that no embedding or vector operations were called
        mock_embedder.return_value.embed_dense_query.assert_not_called()
        mock_embedder.return_value.embed_sparse_query.assert_not_called()
        mock_embedder.return_value.embed_colbert_query.assert_not_called()
        mock_qdrant.return_value.query_hybrid_with_rerank.assert_not_called()
        mock_text_processor.return_value.build_structured_context.assert_not_called()
        mock_llm.return_value.synthesize_answer.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])