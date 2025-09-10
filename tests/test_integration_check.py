#!/usr/bin/env python3
"""
Test script to verify markdown formatting and query processing flow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.text_processing_service import TextProcessingService
from app.services.llm_service import LLMService


def test_text_processing():
    """Test the text processing functionality."""
    print("🧪 Testing Text Processing Service...")
    
    processor = TextProcessingService()
    
    # Test sample raw text (simulating PDF extraction)
    raw_text = """
    **This is a sample document** with various [formatting] and "quoted text".
    
    It contains:
    • Bullet points
    • Multiple lines
    • Some artifacts like page 123 references
    
    The content should be (preserved) for LLM processing.
    """
    
    # Test light cleaning (for LLM input)
    light_cleaned = processor.light_clean_text(raw_text)
    print(f"✅ Light cleaned text:\n{light_cleaned}\n")
    
    # Test aggressive cleaning (for analysis)
    aggressive_cleaned = processor.aggressive_clean_text(raw_text)
    print(f"✅ Aggressive cleaned text:\n{aggressive_cleaned}\n")
    
    # Test context building
    mock_sources = [
        {"text": raw_text, "filename": "document1.pdf"},
        {"text": "Another document with **important** information about Next.js", "filename": "document2.pdf"}
    ]
    
    context = processor.build_structured_context(mock_sources)
    print(f"✅ Built context:\n{context}\n")


def test_llm_service():
    """Test the LLM service functionality."""
    print("🤖 Testing LLM Service...")
    
    llm_service = LLMService()
    
    if llm_service.is_available():
        print("✅ LLM service is available")
        
        # Test prompt building
        question = "What is Next.js?"
        context = "SOURCE 1 - nextjs-docs.pdf:\nNext.js is a React framework for production applications."
        
        prompt = llm_service._build_rag_prompt(question, context)
        print(f"✅ Generated prompt preview:\n{prompt[:200]}...\n")
    else:
        print("⚠️ LLM service not available (missing API key)")
    
    # Test greeting responses
    greeting_response = llm_service._build_greeting_response("Hello", has_documents=True)
    print(f"✅ Greeting response:\n{greeting_response}\n")


def test_integration():
    """Test the integration between services."""
    print("🔗 Testing Service Integration...")
    
    processor = TextProcessingService()
    llm_service = LLMService()
    
    # Simulate a complete flow
    question = "What are the main features of Next.js?"
    mock_sources = [
        {
            "text": "Next.js provides **server-side rendering**, static site generation, and API routes. It's built on React.",
            "filename": "nextjs-overview.pdf"
        },
        {
            "text": "Key features include: automatic code splitting, optimized performance, and built-in CSS support.",
            "filename": "nextjs-features.pdf"
        }
    ]
    
    # Build context
    context = processor.build_structured_context(mock_sources)
    print(f"✅ Context prepared for LLM:\n{context[:200]}...\n")
    
    # Calculate text similarity
    similarity = processor.calculate_text_similarity(question, context)
    print(f"✅ Question-context similarity: {similarity:.3f}\n")
    
    print("✅ Integration test completed successfully!")


if __name__ == "__main__":
    print("🚀 RAG System Component Tests\n")
    print("="*50)
    
    try:
        test_text_processing()
        print("="*50)
        test_llm_service()
        print("="*50)
        test_integration()
        print("="*50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
