"""
Test module for markdown processing and text formatting functionality.
Tests the new modular services for text processing and LLM interactions.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.text_processing_service import TextProcessingService
from app.services.llm_service import LLMService
from app.services.query_service import QueryService
from app.models.query import QueryRequest


class TestTextProcessingService(unittest.TestCase):
    """Test the TextProcessingService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.text_processor = TextProcessingService()
    
    def test_light_clean_text_preserves_structure(self):
        """Test that light cleaning preserves text structure for LLM processing."""
        input_text = """
        This is a **bold** text with some formatting.
        
        • Bullet point 1
        • Bullet point 2
        
        1. Numbered list item
        2. Another item
        
        Some text with    excessive   spaces.
        """
        
        result = self.text_processor.light_clean_text(input_text)
        
        # Should preserve markdown-like formatting
        self.assertIn("**bold**", result)
        self.assertIn("•", result)
        self.assertIn("1.", result)
        
        # Should normalize whitespace
        self.assertNotIn("    excessive   spaces", result)
        self.assertIn("excessive spaces", result)
    
    def test_aggressive_clean_text_removes_formatting(self):
        """Test that aggressive cleaning removes formatting artifacts."""
        input_text = """
        This is a "quoted" text with [brackets] and (parentheses).
        **Bold text** should be cleaned.
        """
        
        result = self.text_processor.aggressive_clean_text(input_text)
        
        # Should remove quotes, brackets, and asterisks
        self.assertNotIn('"', result)
        self.assertNotIn('[', result)
        self.assertNotIn('(', result)
        self.assertNotIn('**', result)
    
    def test_build_structured_context(self):
        """Test building structured context from sources."""
        sources = [
            {
                "text": "This is the first source document with important information.",
                "filename": "doc1.pdf"
            },
            {
                "text": "This is the second source with different content.",
                "filename": "doc2.pdf"
            },
            {
                "text": "",  # Empty text should be skipped
                "filename": "empty.pdf"
            }
        ]
        
        result = self.text_processor.build_structured_context(sources)
        
        # Should include both non-empty sources
        self.assertIn("SOURCE 1 - doc1.pdf", result)
        self.assertIn("SOURCE 2 - doc2.pdf", result)
        
        # Should not include empty source
        self.assertNotIn("empty.pdf", result)
        
        # Should use proper separators
        self.assertIn("---", result)
    
    def test_extract_keywords(self):
        """Test keyword extraction functionality."""
        text = "This is a test document about machine learning and artificial intelligence."
        
        keywords = self.text_processor.extract_keywords(text)
        
        # Should extract meaningful keywords
        self.assertIn("test", keywords)
        self.assertIn("document", keywords)
        self.assertIn("machine", keywords)
        self.assertIn("learning", keywords)
        
        # Should not include stop words
        self.assertNotIn("this", keywords)
        self.assertNotIn("is", keywords)
        self.assertNotIn("a", keywords)
    
    def test_calculate_text_similarity(self):
        """Test text similarity calculation."""
        text1 = "machine learning artificial intelligence"
        text2 = "machine learning deep learning"
        text3 = "completely different topic about cooking"
        
        # Similar texts should have higher similarity
        similarity_high = self.text_processor.calculate_text_similarity(text1, text2)
        similarity_low = self.text_processor.calculate_text_similarity(text1, text3)
        
        self.assertGreater(similarity_high, similarity_low)
        self.assertGreater(similarity_high, 0.0)
        self.assertLessEqual(similarity_high, 1.0)


class TestLLMService(unittest.TestCase):
    """Test the LLMService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_service = LLMService()
    
    def test_build_rag_prompt(self):
        """Test RAG prompt building."""
        question = "What is machine learning?"
        context = "Machine learning is a subset of artificial intelligence."
        
        prompt = self.llm_service._build_rag_prompt(question, context)
        
        # Should include the question and context
        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
        
        # Should include markdown formatting instructions
        self.assertIn("markdown", prompt.lower())
        self.assertIn("**bold text**", prompt)
        self.assertIn("bullet points", prompt)
    
    def test_build_greeting_response(self):
        """Test greeting response generation."""
        # Test greeting with documents
        greeting_response = self.llm_service._build_greeting_response("Hello", has_documents=True)
        self.assertIn("IRA", greeting_response)
        self.assertIn("knowledge base", greeting_response)
        
        # Test greeting without documents
        no_docs_response = self.llm_service._build_greeting_response("Hello", has_documents=False)
        self.assertIn("upload some documents", no_docs_response)
        
        # Test non-greeting with no documents
        question_response = self.llm_service._build_greeting_response("What is AI?", has_documents=False)
        self.assertIn("What is AI?", question_response)
    
    @patch('app.services.llm_service.genai', None)
    def test_synthesize_answer_no_llm(self):
        """Test answer synthesis when LLM is not available."""
        llm_service = LLMService()

        answer, reasoning = llm_service.synthesize_answer("Test question", "Test context")

        self.assertIn("LLM not configured", answer)
        self.assertIn("No reasoning available", reasoning)
        self.assertIn("No reasoning available", reasoning)


class TestQueryServiceIntegration(unittest.TestCase):
    """Test the integrated QueryService with new modular components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the dependencies to avoid actual service calls
        with patch('app.services.query_service.QdrantClientWrapper'), \
             patch('app.services.query_service.ColBERTEmbedder'), \
             patch('app.services.query_service.TextProcessingService'), \
             patch('app.services.query_service.LLMService'):
            self.query_service = QueryService()
    
    def test_build_qdrant_filters_empty_selection(self):
        """Test filter building with empty file selection."""
        filters = {"selected_files": []}
        
        result = self.query_service._build_qdrant_filters(filters)
        
        # Should return a filter that matches nothing
        self.assertIsNotNone(result)
    
    def test_build_qdrant_filters_with_files(self):
        """Test filter building with selected files."""
        filters = {"selected_files": ["doc1.pdf", "doc2.pdf"]}
        
        result = self.query_service._build_qdrant_filters(filters)
        
        # Should return a filter for the selected files
        self.assertIsNotNone(result)
    
    def test_build_qdrant_filters_no_filters(self):
        """Test filter building with no filters."""
        result = self.query_service._build_qdrant_filters(None)
        
        # Should return None for no filters
        self.assertIsNone(result)
    
    @patch('app.services.query_service.QdrantClientWrapper')
    @patch('app.services.query_service.ColBERTEmbedder')
    def test_query_with_no_files_selected(self, mock_embedder, mock_qdrant):
        """Test query handling when no files are selected."""
        # Setup mocks
        mock_embedder.return_value.embed_dense_query.return_value = [0.1, 0.2, 0.3]
        mock_embedder.return_value.embed_sparse_query.return_value = {"token": 0.5}
        mock_embedder.return_value.embed_colbert_query.return_value = [[0.1, 0.2]]
        
        with patch('app.services.query_service.TextProcessingService'), \
             patch('app.services.query_service.LLMService'):
            query_service = QueryService()
        
        # Test with empty file selection
        request = QueryRequest(
            question="Hello",
            filters={"selected_files": []},
            top_k=5
        )
        
        response = query_service.query(request)
        
        # Should return a helpful message about file selection
        self.assertIn("select one or more files", response.answer)


class TestMarkdownFormatting(unittest.TestCase):
    """Test specific markdown formatting scenarios."""
    
    def test_markdown_elements_preservation(self):
        """Test that important markdown elements are preserved through processing."""
        text_processor = TextProcessingService()
        
        # Text with various markdown elements
        input_text = """
        # Main Header
        
        This is **bold text** and this is *italic text*.
        
        ## Subheader
        
        Here's a list:
        • First item
        • Second item
        
        And a numbered list:
        1. First numbered item
        2. Second numbered item
        
        > This is a blockquote
        
        `This is inline code`
        
        ```
        This is a code block
        ```
        """
        
        result = text_processor.light_clean_text(input_text)
        
        # Should preserve markdown syntax
        self.assertIn("#", result)  # Headers
        self.assertIn("**bold text**", result)  # Bold
        self.assertIn("*italic text*", result)  # Italic
        self.assertIn("•", result)  # Bullet points
        self.assertIn("1.", result)  # Numbered lists
        self.assertIn(">", result)  # Blockquotes
        self.assertIn("`", result)  # Code


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
