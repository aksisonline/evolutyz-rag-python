"""
files_service.py
Service for managing file operations (list, delete) in Qdrant.
"""

from qdrant_client import QdrantClient, models
import os

class FilesService:
    """Service for file management operations."""
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self._ensure_filename_index()

    def _ensure_filename_index(self):
        """Ensure that the filename field is indexed for efficient filtering."""
        try:
            # Create index on filename field if it doesn't exist
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            print(f"Created index on 'filename' field for collection {self.collection_name}")
        except Exception as e:
            # Index might already exist, which is fine
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"Index on 'filename' field already exists for collection {self.collection_name}")
            else:
                print(f"Note: Could not create index on 'filename' field: {e}")

    def list_files(self):
        """List all files in the collection."""
        try:
            # Get unique filenames from collection payloads
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            filenames = set()
            for point in response[0]:
                if point.payload and 'filename' in point.payload:
                    filenames.add(point.payload['filename'])
            
            return [{"pathname": filename} for filename in filenames]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def delete_file(self, file_url: str):
        """Delete all points associated with a file."""
        try:
            # Extract filename from URL or use as is
            filename = file_url.split('/')[-1] if '/' in file_url else file_url
            
            print(f"Attempting to delete file: {filename}")
            
            # First, check if any points exist for this file using scroll (more reliable than count)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [{"key": "filename", "match": {"value": filename}}]
                },
                limit=100,  # Get up to 100 points to count
                with_payload=True
            )
            
            points_found = len(scroll_result[0])
            if points_found == 0:
                print(f"No points found for file: {filename}")
                return {"status": "warning", "message": f"No data found for file {filename}."}
            
            # Continue scrolling to get all points if there are more
            total_points = points_found
            next_page_offset = scroll_result[1]
            
            while next_page_offset is not None:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={
                        "must": [{"key": "filename", "match": {"value": filename}}]
                    },
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True
                )
                total_points += len(scroll_result[0])
                next_page_offset = scroll_result[1]
            
            print(f"Found {total_points} points to delete for file: {filename}")
            
            # Delete all points with this filename
            delete_result = self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [{"key": "filename", "match": {"value": filename}}]
                    }
                }
            )
            
            print(f"Delete operation completed for file: {filename}")
            print(f"Delete result: {delete_result}")
            
            return {
                "status": "success", 
                "message": f"File {filename} deleted successfully. Removed {total_points} data points.",
                "deleted_points": total_points
            }
            
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
            return {"status": "error", "message": f"Failed to delete file: {str(e)}"}
