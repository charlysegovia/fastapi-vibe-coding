from pydantic import BaseModel, Field
from typing import Any, Dict

class DocumentUpload(BaseModel):
    filename: str = Field(..., description="PDF file name")
    content: bytes = Field(..., description="Binary content of the PDF")

class ChunkedEmbedding(BaseModel):
    text: str = Field(..., description="Text of the page")
    vector: list[float] = Field(..., description="Embedding vector")
    metadata: Dict[str, Any] = Field(..., description="Metadata: doc_id, chunk_id, page_number, filename") 