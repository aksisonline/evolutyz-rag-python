from fastapi import APIRouter, UploadFile, File
from app.services.ingestion_service import IngestionService
from app.services.pdf_service import PDFIngestionService
from app.models.ingestion import IngestionResponse
import os

router = APIRouter()

# Use consistent collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

ingestion_service = IngestionService()
pdf_service = PDFIngestionService(
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API", ""),
    collection_name=COLLECTION_NAME
)

@router.post("/csv", response_model=IngestionResponse)
def ingest_csv(file: UploadFile = File(...)):
    """Ingest a CSV file and upsert data into Qdrant."""
    return ingestion_service.ingest_csv(file)

@router.post("/pdf")
def ingest_pdf(file: UploadFile = File(...)):
    """Ingest a PDF file and upsert data into Qdrant."""
    # Save uploaded file to temp location
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        pdf_service.ingest_pdf(tmp_path, metadata={"filename": file.filename})
        return {"status": "success", "message": f"PDF {file.filename} ingested."}
    finally:
        os.remove(tmp_path)
