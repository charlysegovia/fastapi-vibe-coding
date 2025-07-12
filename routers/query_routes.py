from fastapi import APIRouter, HTTPException, status, Request
from schemas.query import QueryRequest, QueryResponse, SourceChunk
from services.embedding_service import get_embedding
from services.milvus_service import init_milvus, search_similar
import os
import httpx
import logging
import asyncio

router = APIRouter()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-3.5-turbo"

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: Request, body: QueryRequest):
    """
    Query the RAG system: embed question, search Milvus across all documents, and get answer from OpenAI.
    Returns the answer and the source chunks used, including all filenames.
    """
    logging.info(f"Received query: {body.question}")
    # Embed the question
    try:
        q_vector = await get_embedding(body.question)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding service error: {e}")
    # Search Milvus for similar chunks (with retry)
    retries = 2
    for attempt in range(retries + 1):
        try:
            collection = await init_milvus()
            hits = await search_similar(collection, q_vector, body.top_k)
            break
        except Exception as e:
            logging.error(f"Milvus search failed (attempt {attempt+1}): {e}")
            if attempt == retries:
                raise HTTPException(status_code=500, detail=f"Vector DB error: {e}")
            await asyncio.sleep(2 ** attempt)
    
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")
    
    # Debug: Log the raw hits to see what data we're getting
    logging.info(f"Raw hits from Milvus: {hits}")
    for i, hit in enumerate(hits):
        logging.info(f"Hit {i}: chunk_id={hit.get('chunk_id')}, page_number={hit.get('page_number')}, filename={hit.get('filename')}, text_length={len(hit.get('text', '')) if hit.get('text') else 0}")
    
    # Build context for OpenAI
    context = "\n".join([
        f"[Page {h['page_number']}] {h['text']} (File: {h['filename']})" for h in hits
    ])
    filenames = sorted(set(h["filename"] for h in hits if h["filename"] is not None))
    if not filenames:
        filenames = ["Unknown"]
    prompt = (
        f"You are an assistant. Use only the following document pages to answer.\n"
        f"Documents: {', '.join(filenames)}\n"
        f"Pages:\n{context}\n"
        f"Question: {body.question}\n"
        f"In your answer, specify which documents and pages you used."
    )
    # Call OpenAI chat completion
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"OpenAI chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM service error: {e}")
    # Build response
    sources = []
    for h in hits:
        # Skip hits with None values
        if h["chunk_id"] is None or h["page_number"] is None or h["text"] is None or h["filename"] is None:
            continue
        sources.append(SourceChunk(
            chunk_id=h["chunk_id"],
            page_number=h["page_number"],
            text=h["text"],
            filename=h["filename"]
        ))
    
    if not sources:
        raise HTTPException(status_code=404, detail="No valid source chunks found.")
    
    return QueryResponse(answer=answer, sources=sources) 