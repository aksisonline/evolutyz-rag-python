#!/usr/bin/env python3
"""
create_filename_index.py
Utility script to create the filename index in Qdrant collection.
Run this if you're getting index-related errors when deleting files.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Load environment variables
load_dotenv(".env.local")

def create_filename_index():
    """Create index on filename field for efficient filtering."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API", "")
    collection_name = os.getenv("COLLECTION_NAME", "rag_collection")
    
    print(f"Connecting to Qdrant at {qdrant_url}")
    print(f"Collection: {collection_name}")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    try:
        # Check if collection exists
        collection_info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' found with {collection_info.vectors_count} vectors")
        
        # Create index on filename field
        client.create_payload_index(
            collection_name=collection_name,
            field_name="filename",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("✅ Successfully created index on 'filename' field")
        
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print("✅ Index on 'filename' field already exists")
        else:
            print(f"❌ Error creating index: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("Creating filename index for Qdrant collection...")
    success = create_filename_index()
    if success:
        print("Index creation completed successfully!")
    else:
        print("Index creation failed!")
