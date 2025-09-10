#!/usr/bin/env python3
"""
Test class for Qdrant connection and API validation.
Follows OOPS principles and modular design.
"""

import os
from typing import Optional, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo


class QdrantConnectionTester:
    """
    A class to test Qdrant database connections and validate API credentials.
    Follows OOPS principles with encapsulation and single responsibility.
    """
    
    def __init__(self, env_file: str = ".env.local"):
        """
        Initialize the Qdrant connection tester.
        
        Args:
            env_file (str): Path to the environment file containing Qdrant credentials
        """
        self.env_file = env_file
        self.qdrant_url: Optional[str] = None
        self.qdrant_api: Optional[str] = None
        self.client: Optional[QdrantClient] = None
        self._load_environment_variables()
    
    def _load_environment_variables(self) -> None:
        """Load environment variables from the specified file."""
        load_dotenv(self.env_file)
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api = os.getenv("QDRANT_API")
    
    def _create_client(self) -> None:
        """Create a Qdrant client instance."""
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is not set")
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api
        )
    
    def test_connection(self) -> bool:
        """
        Test the connection to Qdrant and validate API credentials.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self._create_client()
            collections = self.client.get_collections()
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_collections_info(self) -> List[CollectionInfo]:
        """
        Retrieve information about all collections in the Qdrant instance.
        
        Returns:
            List[CollectionInfo]: List of collection information objects
        
        Raises:
            Exception: If connection fails or collections cannot be retrieved
        """
        if not self.client:
            self._create_client()
        
        collections_response = self.client.get_collections()
        return collections_response.collections
    
    def print_connection_status(self) -> None:
        """Print detailed connection status and diagnostics."""
        print(f"Testing connection to: {self.qdrant_url}")
        print(f"Using API key: {self.qdrant_api[:10]}..." if self.qdrant_api else "No API key found")
        
        if self.test_connection():
            print("✅ Connection successful!")
            try:
                collections = self.get_collections_info()
                print(f"Found {len(collections)} collections:")
                for collection in collections:
                    print(f"  - {collection.name}")
            except Exception as e:
                print(f"Could not retrieve collections: {e}")
        else:
            self._print_troubleshooting_tips()
    
    def _print_troubleshooting_tips(self) -> None:
        """Print troubleshooting tips for connection failures."""
        print("\nPossible solutions:")
        print("1. Check if the API key is valid and not expired")
        print("2. Verify the Qdrant URL is correct")
        print("3. Ensure your Qdrant cloud instance is active")
        print("4. Consider using a local Qdrant instance for development")


def main():
    """Main function to run the Qdrant connection test."""
    tester = QdrantConnectionTester()
    tester.print_connection_status()


if __name__ == "__main__":
    main()
