#!/usr/bin/env python3
"""
test_summary_enhancement.py
Test script to demonstrate the enhanced summary functionality with better diversity.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.query_service import QueryService
from app.models.query import QueryRequest

def test_summary_enhancement():
    """Test the enhanced summary functionality with better diversity."""
    query_service = QueryService()
    
    print("🔍 Testing Enhanced Summary Functionality...\n")
    
    print(f"LLM Available: {query_service.is_available()}")
    
    if not query_service.is_available():
        print("❌ LLM not available. Please configure GOOGLE_API_KEY environment variable.")
        return
    
    # Test 1: Regular question (should use normal top_k)
    print("\n" + "="*80)
    print("Test 1: Regular question (normal retrieval)")
    print("="*80)
    question = "What is Next.js routing?"
    print(f"Question: {question}")
    print(f"Is Summary Request: {query_service._is_summary_request(question)}")
    normal_k = query_service._get_enhanced_top_k_for_summary(5)
    print(f"Enhanced top_k for normal query: {normal_k}")
    
    # Test 2: Summary question (should use enhanced top_k)
    print("\n" + "="*80)
    print("Test 2: Summary question (enhanced retrieval)")
    print("="*80)
    question = "Please summarize the key topics covered in the documents"
    print(f"Question: {question}")
    print(f"Is Summary Request: {query_service._is_summary_request(question)}")
    enhanced_k = query_service._get_enhanced_top_k_for_summary(5)
    print(f"Enhanced top_k for summary query: {enhanced_k}")
    
    # Test 3: Different summary variations
    print("\n" + "="*80)
    print("Test 3: Various summary question patterns")
    print("="*80)
    
    summary_questions = [
        "Give me an overview of the documentation",
        "What are the main points covered?",
        "Summarize the content",
        "What information is available in the documents?",
        "Provide a comprehensive view of the topics",
        "What do the documents discuss?",
        "Compile the key features mentioned"
    ]
    
    for q in summary_questions:
        is_summary = query_service._is_summary_request(q)
        enhanced_k = query_service._get_enhanced_top_k_for_summary(5)
        print(f"'{q}' -> Summary: {is_summary}, Enhanced K: {enhanced_k if is_summary else 5}")
    
    # Test 4: Live summary with streaming
    print("\n" + "="*80)
    print("Test 4: Live summary with enhanced streaming")
    print("="*80)
    question = "Summarize the main topics and features covered in the Next.js documentation"
    print(f"Question: {question}")
    print("\nStreaming enhanced summary response:")
    print("-" * 60)
    try:
        chunks = []
        for chunk in query_service.auto_answer_stream(question):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print(f"\n\nTotal chunks received: {len(chunks)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n\n✅ Summary enhancement test completed!")

if __name__ == "__main__":
    test_summary_enhancement()
