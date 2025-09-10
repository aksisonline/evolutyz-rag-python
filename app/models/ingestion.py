from pydantic import BaseModel
from typing import List, Optional

class IngestionResponse(BaseModel):
    success: bool
    message: str
    ingested_rows: int
    errors: Optional[List[str]] = None
