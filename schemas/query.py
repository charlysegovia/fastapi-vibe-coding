from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    filename: Optional[str] = Field(None, description="Document filename to query (optional)")
    top_k: int = Field(3, description="Number of chunks to retrieve")

class SourceChunk(BaseModel):
    chunk_id: str
    page_number: int
    text: str
    filename: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] 