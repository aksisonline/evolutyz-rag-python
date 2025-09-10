#!/usr/bin/env python3
"""
test_file_selection_logic.py
Test script to verify that the query service properly handles file selection.
"""

from app.services.query_service import QueryService
from app.models.query import QueryRequest

def test_file_selection_logic():
    """Test the file selection filtering logic."""
    query_service = QueryService()
    
    print("🧪 Testing file selection logic...\n")
    
    # Test 1: No filters provided
    print("Test 1: No filters provided")
    filters = {}
    qdrant_filters = query_service._build_qdrant_filters(filters)
    print(f"Result: {qdrant_filters}")
    print("Expected: None (search all documents)\n")
    
    # Test 2: Empty selected_files list
    print("Test 2: Empty selected_files list")
    filters = {"selected_files": []}
    qdrant_filters = query_service._build_qdrant_filters(filters)
    print(f"Result: {qdrant_filters}")
    print("Expected: Filter that matches no documents\n")
    
    # Test 3: Some files selected
    print("Test 3: Some files selected")
    filters = {"selected_files": ["file1.pdf", "file2.pdf"]}
    qdrant_filters = query_service._build_qdrant_filters(filters)
    print(f"Result: {qdrant_filters}")
    print("Expected: Filter that matches specified files\n")
    
    # Test 4: Query with no files selected
    print("Test 4: Query with no files selected")
    request = QueryRequest(
        question="Hello, what can you help me with?",
        filters={"selected_files": []},
        top_k=5
    )
    
    # Test the streaming response
    print("Streaming response:")
    response_chunks = []
    try:
        for chunk in query_service.stream_answer(request):
            response_chunks.append(chunk)
            print(f"Chunk: {chunk}")
    except Exception as e:
        print(f"Error: {e}")
    
    full_response = "".join(response_chunks)
    print(f"\nFull response: {full_response}")
    print("Expected: Message about selecting files\n")

if __name__ == "__main__":
    test_file_selection_logic()
