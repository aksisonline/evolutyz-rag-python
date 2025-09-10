
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import asyncio
from app.services.query_service import QueryService
# from app.services.pdf_query_service import PDFQueryService  # Disabled to avoid memory issues
from app.models.query import QueryRequest, QueryResponse
import os

router = APIRouter()

query_service = QueryService()
# pdf_query_service = PDFQueryService(  # Disabled to avoid memory issues at startup
#     qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
#     qdrant_api_key=os.getenv("QDRANT_API", ""),
#     collection_name=os.getenv("COLLECTION_NAME", "pdf_collection")
# )

@router.post("/", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Query the RAG pipeline."""
    return query_service.query(request)

@router.post("/pdf")
def query_pdf(query: str, search_limit: int = 10, prefetch_limit: int = 100):
    """Query the PDF pipeline - Currently disabled to avoid memory issues."""
    return {"error": "PDF query service is currently disabled to avoid memory issues. Use the main /query endpoint instead."}

# @router.post("/pdf")
# def query_pdf(query: str, search_limit: int = 10, prefetch_limit: int = 100):
#     """Query the PDF pipeline."""
#     response = pdf_query_service.query_pdf(query, search_limit, prefetch_limit)
#     return {"results": [point.payload for point in response.points]}

@router.post("/stream")
async def stream_rag(request: Request):
    body = await request.json()
    query_req = QueryRequest(**body)

    async def token_stream():
        # Stream chunks; ensure each newline becomes a separate SSE data line per spec
        for chunk in query_service.stream_answer(query_req):
            if chunk is None:
                continue
            # Normalize Windows line endings
            chunk = chunk.replace('\r\n', '\n')
            lines = chunk.split('\n')
            for i, line in enumerate(lines):
                # Preserve empty lines explicitly
                yield f"data: {line}\n"
            # Event terminator
            yield "\n"
            await asyncio.sleep(0)  # Yield control to event loop

    return StreamingResponse(token_stream(), media_type="text/event-stream")
