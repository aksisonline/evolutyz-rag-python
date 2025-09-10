#!/usr/bin/env python3
"""
test_delete_functionality.py
Test script to verify the file deletion functionality works correctly.
"""

import os
from dotenv import load_dotenv
from app.services.files_service import FilesService

# Load environment variables
load_dotenv(".env.local")

def test_delete_functionality():
    """Test the file deletion functionality."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API", "")
    collection_name = os.getenv("COLLECTION_NAME", "rag_collection")
    
    print(f"Testing delete functionality...")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: {collection_name}")
    
    # Initialize files service
    files_service = FilesService(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name
    )
    
    # List current files
    print("\n📁 Current files in collection:")
    files = files_service.list_files()
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file['pathname']}")
    
    if not files:
        print("  No files found in collection.")
        return
    
    # Test delete with a non-existent file
    print("\n🧪 Testing delete with non-existent file...")
    result = files_service.delete_file("non_existent_file.pdf")
    print(f"Result: {result}")
    
    print("\n✅ Delete functionality test completed!")

if __name__ == "__main__":
    test_delete_functionality()
