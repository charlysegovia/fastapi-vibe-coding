import os
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from typing import List, Dict, Any
import logging
from pymilvus.exceptions import MilvusException

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = "rag_documents"
EMBEDDING_DIM = 1536

async def init_milvus():
    """Initialize the connection and collection in Milvus."""
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    if not utility.has_collection(COLLECTION_NAME):
        logging.info("Creating Milvus collection...")
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields, description="RAG PDF chunks")
        Collection(COLLECTION_NAME, schema)
    collection = Collection(COLLECTION_NAME)
    # Ensure index exists on embedding field
    if not any(idx.field_name == "embedding" for idx in collection.indexes):
        logging.info("Creating index on embedding field...")
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128}
            }
        )
    # Always load the collection (idempotent)
    try:
        collection.load()
    except MilvusException as e:
        logging.error(f"Failed to load collection: {e}")
        raise
    return collection

async def insert_embeddings(collection: Collection, embeddings: List[Dict[str, Any]]):
    """Insert embeddings and metadata into the collection."""
    data = [
        [e["chunk_id"] for e in embeddings],
        [e["doc_id"] for e in embeddings],
        [e["filename"] for e in embeddings],
        [e["page_number"] for e in embeddings],
        [e["text"] for e in embeddings],
        [e["vector"] for e in embeddings],
    ]
    collection.insert(data)
    collection.flush()

async def search_similar(collection: Collection, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Search for the top_k most relevant chunks across all documents."""
    try:
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 8}},
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "filename", "page_number", "text"]
        )
        hits = results[0]
        return [
            {
                "chunk_id": hit.entity.get("chunk_id"),
                "page_number": hit.entity.get("page_number"),
                "text": hit.entity.get("text"),
                "filename": hit.entity.get("filename"),
                "score": hit.distance
            }
            for hit in hits
        ]
    except MilvusException as e:
        logging.error(f"Milvus search failed: {e}")
        raise

async def milvus_status() -> Dict[str, str]:
    """Check Milvus connection and collection status."""
    try:
        connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
        status = {"connected": "true"}
        if utility.has_collection(COLLECTION_NAME):
            status["collection"] = "ready"
        else:
            status["collection"] = "missing"
        return status
    except Exception as e:
        return {"connected": "false", "error": str(e)} 