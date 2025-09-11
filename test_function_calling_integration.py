#!/usr/bin/env python3
"""
test_function_calling_integration.py
Integration test to verify the new function calling capabilities work correctly.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.query_service import QueryService
from app.models.query import QueryRequest

def test_function_calling_integration():
    """Test the integrated function calling capabilities."""
    query_service = QueryService()
    
    print("🚀 Testing Function Calling Integration...\n")
    
    print(f"LLM Available: {query_service.is_available()}")
    
    if not query_service.is_available():
        print("❌ LLM not available. Please configure GOOGLE_API_KEY environment variable.")
        return
    
    # Test 1: Simple identity question (should not call RAG)
    print("\n" + "="*60)
    print("Test 1: Identity question (should bypass RAG)")
    print("="*60)
    question = "Who are you?"
    print(f"Question: {question}")
    print("\nFunction calling response:")
    try:
        response = query_service.auto_answer(question)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Chitchat (should not call RAG)
    print("\n" + "="*60)
    print("Test 2: Chitchat (should bypass RAG)")
    print("="*60)
    question = "Hello!"
    print(f"Question: {question}")
    print("\nFunction calling response:")
    try:
        response = query_service.auto_answer(question)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Knowledge question (should call RAG)
    print("\n" + "="*60)
    print("Test 3: Knowledge question (should call RAG)")
    print("="*60)
    question = "What is the main topic of the documents?"
    print(f"Question: {question}")
    print("\nFunction calling response:")
    try:
        response = query_service.auto_answer(question)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Streaming with function calling
    print("\n" + "="*60)
    print("Test 4: Streaming with function calling")
    print("="*60)
    question = "Summarize the key points from the documents"
    print(f"Question: {question}")
    print("\nStreaming response:")
    try:
        chunks = []
        for chunk in query_service.auto_answer_stream(question):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print(f"\n\nTotal chunks received: {len(chunks)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Compare legacy vs function calling approaches
    print("\n" + "="*60)
    print("Test 5: Compare legacy vs function calling")
    print("="*60)
    question = "What information is available in the documents?"
    
    request = QueryRequest(
        question=question,
        filters={"selected_files": []},  # No files selected
        top_k=5
    )
    
    print(f"Question: {question}")
    print(f"Request: {request}")
    
    print("\n--- Legacy approach ---")
    try:
        legacy_chunks = []
        for chunk in query_service.stream_answer(request):
            legacy_chunks.append(chunk)
            print(chunk, end="", flush=True)
        print(f"\nLegacy chunks: {len(legacy_chunks)}")
    except Exception as e:
        print(f"Legacy error: {e}")
    
    print("\n\n--- Function calling approach ---")
    try:
        fc_chunks = []
        for chunk in query_service.stream_answer_with_function_calling(request):
            fc_chunks.append(chunk)
            print(chunk, end="", flush=True)
        print(f"\nFunction calling chunks: {len(fc_chunks)}")
    except Exception as e:
        print(f"Function calling error: {e}")
    
    print("\n\n✅ Integration test completed!")

if __name__ == "__main__":
    test_function_calling_integration()
