from fastapi import APIRouter, HTTPException, status, Request
from schemas.query import QueryRequest, QueryResponse, SourceChunk
from main_working import get_load_data_service, get_service_status
import os
import httpx
import logging
import asyncio
from typing import List, Dict

router = APIRouter()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-3.5-turbo"

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: Request, body: QueryRequest):
    """
    Query the RAG system using hybrid search (dense + sparse).
    Performs semantic search across all processed repositories and generates an answer using OpenAI.
    """
    logging.info(f"Received query: {body.question}")
    
    # Perform hybrid search
    try:
        status = get_service_status()
        if not status["service_ready"]:
            raise HTTPException(status_code=503, detail=f"Service not ready. Status: {status}")
        
        service = get_load_data_service()
        search_results = await service.search_hybrid(
            query=body.question, 
            top_k=body.top_k,
            enable_reranking=body.enable_reranking,
            rerank_top_k=body.rerank_top_k,
            show_scores=body.show_scores,
            show_justification=body.show_justification
        )
        logging.info(f"Hybrid search returned {len(search_results['results'])} results")
        if search_results.get('re_ranking', {}).get('enabled'):
            logging.info(f"Re-ranking: {search_results['re_ranking']['successful']}, evaluated: {search_results['re_ranking'].get('evaluated_count', 0)}")
    except Exception as e:
        logging.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    
    if not search_results['results']:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")
    
    # Prepare context from search results
    context_chunks = []
    for result in search_results['results']:
        context_chunks.append({
            "text": result['text'],
            "repo_name": result['repo_name'],
            "file_path": result['file_path'],
            "chunk_type": result['chunk_type'],
            "file_language": result['file_language'],
            "author": result['author'],
            "score": result['score'],
            "search_type": result['search_type'],
            "relevance_score": result.get('relevance_score'),
            "justification": result.get('justification'),
            "original_rank": result.get('original_rank')
        })
    
    # Generate answer using OpenAI
    try:
        answer = await _generate_answer_with_openai(body.question, context_chunks)
    except Exception as e:
        logging.error(f"OpenAI answer generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e}")
    
    # Prepare response
    sources = []
    for chunk in context_chunks:
        sources.append(SourceChunk(
            text=chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
            repo_name=chunk['repo_name'],
            file_path=chunk['file_path'],
            chunk_type=chunk['chunk_type'],
            file_language=chunk['file_language'],
            author=chunk['author'],
            score=chunk['score'],
            search_type=chunk['search_type'],
            relevance_score=chunk.get('relevance_score'),
            justification=chunk.get('justification'),
            original_rank=chunk.get('original_rank')
        ))
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        search_stats={
            "dense_results": search_results['dense_count'],
            "sparse_results": search_results['sparse_count'],
            "total_results": search_results['total_results'],
            "re_ranking": search_results.get('re_ranking', {})
        }
    )

async def _generate_answer_with_openai(question: str, context_chunks: List[Dict]) -> str:
    """Generate an answer using OpenAI based on the retrieved context."""
    # Prepare context
    context_text = "\n\n".join([
        f"Source: {chunk['repo_name']}/{chunk['file_path']} ({chunk['file_language']})\n"
        f"Author: {chunk['author']}\n"
        f"Content: {chunk['text']}"
        for chunk in context_chunks
    ])
    
    # Create prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from GitHub repositories.

Context:
{context_text}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so. Cite specific files and code snippets when relevant.

Answer:"""
    
    # Call OpenAI
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on GitHub repository content."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()

@router.get("/search-stats")
async def get_search_stats(request: Request) -> Dict:
    """
    Get statistics about the search system.
    Returns information about processed repositories and available data.
    """
    try:
        status = get_service_status()
        if not status["service_ready"]:
            raise HTTPException(status_code=503, detail=f"Service not ready. Status: {status}")
        
        # Get all repositories
        service = get_load_data_service()
        repos = await service.list_repositories()
        
        # Get stats for each repository
        repo_stats = []
        total_dense_chunks = 0
        total_sparse_chunks = 0
        
        for repo in repos:
            try:
                repo_status = await service.get_processing_status(repo['name'])
                repo_stats.append({
                    "name": repo['name'],
                    "description": repo['description'],
                    "language": repo['language'],
                    "dense_chunks": status.get('dense_chunks', 0),
                    "sparse_chunks": status.get('sparse_chunks', 0),
                    "total_chunks": status.get('total_chunks', 0)
                })
                total_dense_chunks += status.get('dense_chunks', 0)
                total_sparse_chunks += status.get('sparse_chunks', 0)
            except Exception as e:
                logging.warning(f"Could not get stats for {repo['name']}: {e}")
                repo_stats.append({
                    "name": repo['name'],
                    "description": repo['description'],
                    "language": repo['language'],
                    "dense_chunks": 0,
                    "sparse_chunks": 0,
                    "total_chunks": 0,
                    "error": str(e)
                })
        
        return {
            "total_repositories": len(repos),
            "processed_repositories": len([r for r in repo_stats if r.get('total_chunks', 0) > 0]),
            "total_dense_chunks": total_dense_chunks,
            "total_sparse_chunks": total_sparse_chunks,
            "total_chunks": total_dense_chunks + total_sparse_chunks,
            "repositories": repo_stats
        }
        
    except Exception as e:
        logging.error(f"Failed to get search stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get search stats: {str(e)}") 