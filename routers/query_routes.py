from fastapi import APIRouter, HTTPException, status, Request
from schemas.query import QueryRequest, QueryResponse, SourceChunk
from services.embedding_service import get_embedding
from services.milvus_service import init_milvus, search_similar
import os
import httpx
import logging

router = APIRouter()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-3.5-turbo"

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: Request, body: QueryRequest):
    """
    Query a document using RAG: embed question, search Milvus, and get answer from OpenAI.
    Returns the answer and the source chunks used.
    """
    logging.info(f"Received query for file: {body.filename}")
    # Embed the question
    try:
        q_vector = await get_embedding(body.question)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding service error.")
    # Search Milvus for similar chunks
    try:
        collection = await init_milvus()
        hits = await search_similar(collection, q_vector, body.filename, body.top_k)
    except Exception as e:
        logging.error(f"Milvus search failed: {e}")
        raise HTTPException(status_code=500, detail="Vector DB error.")
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")
    # Build context for OpenAI
    context = "\n".join([
        f"[Page {h['page_number']}] {h['text']}" for h in hits
    ])
    prompt = (
        f"You are an assistant. Use only the following document pages to answer.\n"
        f"Document: {body.filename}\n"
        f"Pages:\n{context}\n"
        f"Question: {body.question}\n"
        f"In your answer, specify which document and pages you used."
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
        raise HTTPException(status_code=500, detail="LLM service error.")
    # Build response
    sources = [
        SourceChunk(
            chunk_id=h["chunk_id"],
            page_number=h["page_number"],
            text=h["text"],
            filename=h["filename"]
        ) for h in hits
    ]
    return QueryResponse(answer=answer, sources=sources) 