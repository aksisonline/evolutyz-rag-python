from pydantic import BaseModel
from typing import List, Optional

class IngestionResponse(BaseModel):
    success: bool
    message: str
    ingested_rows: int  # number of original CSV rows processed
    segments: Optional[int] = None  # total number of segments actually ingested (may exceed rows)
    errors: Optional[List[str]] = None
