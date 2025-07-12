import os
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from typing import List, Dict, Any
import logging

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
    return Collection(COLLECTION_NAME)

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

async def search_similar(collection: Collection, query_vector: List[float], filename: str, top_k: int) -> List[Dict[str, Any]]:
    """Search for the top_k most similar chunks for a given document."""
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=top_k,
        expr=f"filename == '{filename}'"
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