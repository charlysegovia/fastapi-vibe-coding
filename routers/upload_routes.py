from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from schemas.upload import DocumentUpload, ChunkedEmbedding
from utils.pdf_utils import extract_text_by_page
from services.embedding_service import get_embedding
from services.milvus_service import init_milvus, insert_embeddings
import uuid
import logging
from pymilvus.exceptions import MilvusException

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@router.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF file, split it by page, generate embeddings, and store them in the shared Milvus collection.
    Returns the filename, doc_id, and number of pages processed.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB).")
    filename = file.filename
    doc_id = str(uuid.uuid4())
    logging.info(f"Processing upload: {filename} (doc_id={doc_id})")
    # Extract text by page
    try:
        pages = extract_text_by_page(contents)
    except Exception as e:
        logging.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to parse PDF.")
    # Generate embeddings for each page
    embeddings = []
    for i, text in enumerate(pages):
        if not text.strip():
            continue
        try:
            vector = await get_embedding(text)
        except Exception as e:
            logging.error(f"Embedding failed on page {i+1}: {e}")
            raise HTTPException(status_code=500, detail="Embedding service error.")
        chunk_id = str(uuid.uuid4())
        embeddings.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "filename": filename,
            "page_number": i + 1,
            "text": text,
            "vector": vector
        })
    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid text found in PDF.")
    # Store in Milvus (shared collection)
    try:
        collection = await init_milvus()
        await insert_embeddings(collection, embeddings)
    except MilvusException as e:
        logging.error(f"Milvus insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector DB error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during Milvus insert: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    logging.info(f"Upload complete: {filename} (pages={len(embeddings)})")
    return {"filename": filename, "doc_id": doc_id, "pages": len(embeddings)} 