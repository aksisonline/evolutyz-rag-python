"""
files.py
API endpoints for file management (list, delete, upload) - Used by frontend.
"""

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.services.files_service import FilesService
from app.services.pdf_service import PDFIngestionService
from app.services.ingestion_service import IngestionService
import os
import tempfile

router = APIRouter()

# Use consistent collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

files_service = FilesService(
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API", ""),
    collection_name=COLLECTION_NAME
)

pdf_service = PDFIngestionService(
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API", ""),
    collection_name=COLLECTION_NAME
)

ingestion_service = IngestionService()

@router.get("/list")
def list_files():
    """List all uploaded files."""
    return files_service.list_files()

@router.delete("/delete")
def delete_file(fileurl: str):
    """Delete a file and all its associated data points from the vector database."""
    try:
        if not fileurl:
            raise HTTPException(status_code=400, detail="File URL parameter is required")
        
        print(f"Delete request received for file: {fileurl}")
        result = files_service.delete_file(fileurl)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        elif result["status"] == "warning":
            # Return 200 with warning message for files that don't exist
            return result
        
        print(f"Successfully deleted file: {fileurl}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in delete_file endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/upload")
async def upload_file(request: Request, filename: str):
    """Upload and ingest a file."""
    if not filename.lower().endswith(('.pdf', '.csv')):
        return {"status": "error", "message": "Only PDF and CSV files are supported"}
    
    # Read the raw file data from request body
    file_data = await request.body()
    
    if not file_data:
        raise HTTPException(status_code=400, detail="No file data received")
    
    # Determine file extension for temp file
    file_ext = ".pdf" if filename.lower().endswith('.pdf') else ".csv"
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name
    
    try:
        if filename.lower().endswith('.pdf'):
            pdf_service.ingest_pdf(tmp_path, metadata={"filename": filename})
            return {"status": "success", "message": f"PDF {filename} uploaded and ingested successfully."}
        else:
            # Handle CSV files using IngestionService
            from fastapi import UploadFile
            from io import BytesIO
            
            # Create UploadFile object from the file data
            csv_file = UploadFile(
                filename=filename,
                file=BytesIO(file_data)
            )
            
            result = ingestion_service.ingest_csv(csv_file)
            if result.success:
                return {"status": "success", "message": f"CSV {filename} uploaded and ingested successfully. Rows processed: {result.ingested_rows}"}
            else:
                return {"status": "error", "message": f"CSV ingestion failed: {result.message}"}
                
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {str(e)}"}
    finally:
        os.remove(tmp_path)
