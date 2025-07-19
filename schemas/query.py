from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    enable_reranking: bool = Field(default=True, description="Enable ChatGPT re-ranking of results")
    rerank_top_k: int = Field(default=5, ge=1, le=10, description="Number of top results after re-ranking")
    show_scores: bool = Field(default=True, description="Show relevance scores in results")
    show_justification: bool = Field(default=True, description="Show justification for each result")

class SourceChunk(BaseModel):
    text: str = Field(..., description="The text content of the chunk")
    repo_name: str = Field(..., description="Repository name")
    file_path: str = Field(..., description="File path within the repository")
    chunk_type: str = Field(..., description="Type of chunk (paragraph, sentence_group, file, section)")
    file_language: str = Field(..., description="Programming language of the file")
    author: Optional[str] = Field(None, description="Author of the file")
    score: float = Field(..., description="Similarity score")
    search_type: str = Field(..., description="Type of search that found this result (dense or sparse)")
    relevance_score: Optional[float] = Field(None, description="ChatGPT relevance score (1-10)")
    justification: Optional[str] = Field(None, description="ChatGPT justification for the relevance score")
    original_rank: Optional[int] = Field(None, description="Original rank before re-ranking")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[SourceChunk] = Field(..., description="Source chunks used to generate the answer")
    search_stats: Dict[str, Any] = Field(..., description="Statistics about the search results") 