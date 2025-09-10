#!/usr/bin/env python3
"""
Test script to verify that the RAG system now provides concise responses.
This script can be used to test the modified query service.
"""

import requests
import json

def test_concise_response():
    """Test the RAG API with a sample query to check response length"""
    
    # API endpoint (adjust if your server runs on a different port)
    api_url = "http://localhost:3001/query/"
    
    # Sample query that typically generates long responses
    test_query = {
        "question": "Explain Abhiram's educational background and key skills",
        "top_k": 5,
        "filters": {
            "selected_files": []  # Empty list means all files
        }
    }
    
    try:
        print("Testing concise response generation...")
        print(f"Query: {test_query['question']}")
        print("-" * 50)
        
        # Make API request
        response = requests.post(api_url, json=test_query)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Count words in the response
            word_count = len(answer.split())
            
            print(f"Response (Word count: {word_count}):")
            print(answer)
            print("-" * 50)
            print(f"Sources found: {len(sources)}")
            
            # Check if response is concise (under 200 words as per our guidelines)
            if word_count <= 200:
                print("✅ SUCCESS: Response is concise!")
            else:
                print("⚠️  WARNING: Response might be too long")
                
        else:
            print(f"❌ ERROR: API returned status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API. Make sure the server is running on localhost:3001")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_concise_response()
