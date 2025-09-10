from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env.local file
load_dotenv(".env.local")

from app.api import ingestion, query, files

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for modular endpoints
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(files.router, prefix="/files", tags=["files"])

@app.get("/")
def root():
    return {"message": "Qdrant RAG Pipeline is running."}
