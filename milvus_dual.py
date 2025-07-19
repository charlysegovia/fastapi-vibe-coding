import os
import logging
from typing import List, Dict, Any, Optional
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from pymilvus.exceptions import MilvusException
from services.embedding_service import get_embedding
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

class MilvusDualService:
    """Service for managing dual Milvus collections (dense and sparse)."""
    
    def __init__(self):
        self.milvus_uri = os.getenv("MILVUS_URI")
        self.milvus_token = os.getenv("MILVUS_TOKEN")
        self.dense_collection_name = "rag_documents_dense"
        self.sparse_collection_name = "rag_documents_sparse"
        self.embedding_dim = 1536
        
        # TF-IDF vectorizer for sparse embeddings - more lenient for short texts
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=0.001,  # Much more lenient for short texts
            max_df=0.999,  # More lenient upper bound
            lowercase=True,
            analyzer='word'
        )
        self.tfidf_fitted = False
        self.tfidf_dimension = 1000  # Store the expected dimension
        
    async def initialize(self):
        """Initialize Milvus connection and create collections."""
        try:
            connections.connect(uri=self.milvus_uri, token=self.milvus_token)
            await self._create_dense_collection()
            await self._create_sparse_collection()
            logging.info("Milvus dual collections initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Milvus: {e}")
            raise
    
    async def _create_dense_collection(self):
        """Create the dense collection for semantic embeddings."""
        if not utility.has_collection(self.dense_collection_name):
            logging.info("Creating dense collection...")
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="branch", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="commit_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="last_modified", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="file_size", dtype=DataType.INT64),
                FieldSchema(name="chunking_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="file_language", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Dense embeddings for RAG")
            Collection(self.dense_collection_name, schema)
        
        collection = Collection(self.dense_collection_name)
        
        # Create index if it doesn't exist
        if not any(idx.field_name == "embedding" for idx in collection.indexes):
            logging.info("Creating index on dense embedding field...")
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 128}
                }
            )
        
        try:
            collection.load()
        except MilvusException as e:
            logging.error(f"Failed to load dense collection: {e}")
            raise
    
    async def _create_sparse_collection(self):
        """Create the sparse collection for TF-IDF vectors."""
        if not utility.has_collection(self.sparse_collection_name):
            logging.info("Creating sparse collection...")
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="branch", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="commit_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="last_modified", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="file_size", dtype=DataType.INT64),
                FieldSchema(name="chunking_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="file_language", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="tfidf_vector", dtype=DataType.FLOAT_VECTOR, dim=1000),  # TF-IDF dimension
            ]
            schema = CollectionSchema(fields, description="Sparse embeddings for RAG")
            Collection(self.sparse_collection_name, schema)
        
        collection = Collection(self.sparse_collection_name)
        
        # Create index if it doesn't exist
        if not any(idx.field_name == "tfidf_vector" for idx in collection.indexes):
            logging.info("Creating index on sparse tfidf_vector field...")
            collection.create_index(
                field_name="tfidf_vector",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 128}
                }
            )
        
        try:
            collection.load()
        except MilvusException as e:
            logging.error(f"Failed to load sparse collection: {e}")
            raise
    
    async def insert_dense_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert dense chunks into the dense collection."""
        if not chunks:
            return
        
        # Check if collection exists
        if not utility.has_collection(self.dense_collection_name):
            logging.warning(f"Collection {self.dense_collection_name} does not exist. Creating it...")
            await self._create_dense_collection()
        
        collection = Collection(self.dense_collection_name)
        
        # Generate embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = []
        
        for text in texts:
            try:
                embedding = await get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Failed to generate embedding: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.embedding_dim)
        
        # Prepare data for insertion
        data = [
            [chunk['chunk_id'] for chunk in chunks],
            [chunk['metadata']['repo_name'] for chunk in chunks],
            [chunk['metadata']['file_path'] for chunk in chunks],
            [chunk['metadata']['branch'] for chunk in chunks],
            [chunk['metadata']['commit_hash'] for chunk in chunks],
            [chunk['metadata']['author'] for chunk in chunks],
            [chunk['metadata']['last_modified'] for chunk in chunks],
            [chunk['metadata']['file_size'] for chunk in chunks],
            [chunk['metadata']['chunking_type'] for chunk in chunks],
            [chunk['metadata']['file_language'] for chunk in chunks],
            [chunk['chunk_type'] for chunk in chunks],
            [chunk['chunk_index'] for chunk in chunks],
            [chunk['text'] for chunk in chunks],
            embeddings
        ]
        
        collection.insert(data)
        collection.flush()
        logging.info(f"Inserted {len(chunks)} dense chunks")
    
    async def insert_sparse_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert sparse chunks into the sparse collection."""
        if not chunks:
            return
        
        # Check if collection exists
        if not utility.has_collection(self.sparse_collection_name):
            logging.warning(f"Collection {self.sparse_collection_name} does not exist. Creating it...")
            await self._create_sparse_collection()
        
        collection = Collection(self.sparse_collection_name)
        
        # Generate TF-IDF vectors for all chunks
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            # Check if we have enough text for TF-IDF
            total_text = ' '.join(texts)
            logging.info(f"Processing {len(texts)} texts with total length: {len(total_text)}")
            
            if len(total_text.strip()) < 5:  # Even shorter threshold
                raise ValueError("Text too short for TF-IDF processing")
            
            if not self.tfidf_fitted:
                # Fit the vectorizer on the first batch
                logging.info("Fitting TF-IDF vectorizer...")
                self.tfidf_vectorizer.fit(texts)
                self.tfidf_fitted = True
                # Update dimension based on actual vocabulary size
                self.tfidf_dimension = len(self.tfidf_vectorizer.vocabulary_)
                logging.info(f"TF-IDF vectorizer fitted with {self.tfidf_dimension} features")
                logging.info(f"Vocabulary: {list(self.tfidf_vectorizer.vocabulary_.keys())[:10]}...")
            
            tfidf_vectors = self.tfidf_vectorizer.transform(texts).toarray()
            
            # Ensure all vectors have exactly 1000 dimensions for Milvus
            padded_vectors = []
            for vector in tfidf_vectors:
                # Always ensure 1000 dimensions regardless of TF-IDF dimension
                if len(vector) < 1000:
                    # Pad with zeros if vector is shorter
                    padded_vector = np.zeros(1000)
                    padded_vector[:len(vector)] = vector
                    padded_vectors.append(padded_vector.tolist())
                elif len(vector) > 1000:
                    # Truncate if vector is longer
                    padded_vectors.append(vector[:1000].tolist())
                else:
                    padded_vectors.append(vector.tolist())
            
            # Log vector dimensions before insertion
            for i, vector in enumerate(padded_vectors):
                logging.info(f"TF-IDF Vector {i} dimension: {len(vector)}")
            
            # Prepare data for insertion
            data = [
                [chunk['chunk_id'] for chunk in chunks],
                [chunk['metadata']['repo_name'] for chunk in chunks],
                [chunk['metadata']['file_path'] for chunk in chunks],
                [chunk['metadata']['branch'] for chunk in chunks],
                [chunk['metadata']['commit_hash'] for chunk in chunks],
                [chunk['metadata']['author'] for chunk in chunks],
                [chunk['metadata']['last_modified'] for chunk in chunks],
                [chunk['metadata']['file_size'] for chunk in chunks],
                [chunk['metadata']['chunking_type'] for chunk in chunks],
                [chunk['metadata']['file_language'] for chunk in chunks],
                [chunk['chunk_type'] for chunk in chunks],
                [chunk['chunk_index'] for chunk in chunks],
                [chunk['text'] for chunk in chunks],
                padded_vectors
            ]
            
            collection.insert(data)
            collection.flush()
            logging.info(f"Inserted {len(chunks)} sparse chunks with dimension {self.tfidf_dimension}")
            
        except Exception as e:
            logging.error(f"Failed to insert sparse chunks: {e}")
            # If TF-IDF fails, try with a simpler approach
            logging.info("Falling back to simple sparse vectors")
            logging.info(f"TF-IDF error details: {type(e).__name__}: {str(e)}")
            await self._insert_simple_sparse_chunks(chunks)
    
    async def _insert_simple_sparse_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert chunks with simple sparse vectors as fallback."""
        # Check if collection exists
        if not utility.has_collection(self.sparse_collection_name):
            logging.warning(f"Collection {self.sparse_collection_name} does not exist. Creating it...")
            await self._create_sparse_collection()
        
        collection = Collection(self.sparse_collection_name)
        
        # Create simple sparse vectors (bag of words approach)
        simple_vectors = []
        for chunk in chunks:
            # Create a simple vector based on word frequency
            text = chunk['text'].lower()
            words = text.split()
            word_freq = {}
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Create a 1000-dimensional vector
            vector = np.zeros(1000)
            for i, (word, freq) in enumerate(list(word_freq.items())[:1000]):
                vector[i] = freq
            
            # Normalize the vector
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            # Ensure the vector is exactly 1000 dimensions (double-check)
            if len(vector) != 1000:
                if len(vector) > 1000:
                    vector = vector[:1000]
                else:
                    # Pad with zeros if shorter
                    padded = np.zeros(1000)
                    padded[:len(vector)] = vector
                    vector = padded
            
            # Final verification - ensure exactly 1000 dimensions
            assert len(vector) == 1000, f"Vector dimension mismatch: {len(vector)} != 1000"
            
            simple_vectors.append(vector.tolist())
        
        # Verify all vectors have correct dimension
        for i, vector in enumerate(simple_vectors):
            if len(vector) != 1000:
                logging.error(f"Vector {i} has wrong dimension: {len(vector)} != 1000")
                # Fix the vector
                if len(vector) > 1000:
                    simple_vectors[i] = vector[:1000]
                else:
                    padded = [0.0] * 1000
                    padded[:len(vector)] = vector
                    simple_vectors[i] = padded
        
        # Prepare data for insertion
        data = [
            [chunk['chunk_id'] for chunk in chunks],
            [chunk['metadata']['repo_name'] for chunk in chunks],
            [chunk['metadata']['file_path'] for chunk in chunks],
            [chunk['metadata']['branch'] for chunk in chunks],
            [chunk['metadata']['commit_hash'] for chunk in chunks],
            [chunk['metadata']['author'] for chunk in chunks],
            [chunk['metadata']['last_modified'] for chunk in chunks],
            [chunk['metadata']['file_size'] for chunk in chunks],
            [chunk['metadata']['chunking_type'] for chunk in chunks],
            [chunk['metadata']['file_language'] for chunk in chunks],
            [chunk['chunk_type'] for chunk in chunks],
            [chunk['chunk_index'] for chunk in chunks],
            [chunk['text'] for chunk in chunks],
            simple_vectors
        ]
        
        # Log vector dimensions before insertion
        for i, vector in enumerate(simple_vectors):
            logging.info(f"Vector {i} dimension: {len(vector)}")
        
        collection.insert(data)
        collection.flush()
        logging.info(f"Inserted {len(chunks)} simple sparse chunks")
    
    async def search_hybrid(self, query: str, top_k: int = 10, enable_reranking: bool = True, 
                          rerank_top_k: int = 5, show_scores: bool = True, 
                          show_justification: bool = True) -> Dict[str, Any]:
        """Perform hybrid search using both dense and sparse collections with optional re-ranking."""
        try:
            # Generate dense embedding for query
            query_embedding = await get_embedding(query)
            
            # Generate sparse embedding for query
            if self.tfidf_fitted:
                query_tfidf = self.tfidf_vectorizer.transform([query]).toarray()[0]
                # Ensure query vector has 1000 dimensions
                if len(query_tfidf) < 1000:
                    padded = np.zeros(1000)
                    padded[:len(query_tfidf)] = query_tfidf
                    query_tfidf = padded
                elif len(query_tfidf) > 1000:
                    query_tfidf = query_tfidf[:1000]
            else:
                query_tfidf = np.zeros(1000)
            
            # Search dense collection
            dense_results = await self._search_dense(query_embedding, top_k)
            
            # Search sparse collection
            sparse_results = await self._search_sparse(query_tfidf, top_k)
            
            # Combine and rank results
            combined_results = self._combine_results(dense_results, sparse_results, top_k)
            
            # Apply re-ranking if enabled
            if enable_reranking and combined_results:
                try:
                    from reranking_service import ReRankingService
                    reranking_service = ReRankingService()
                    
                    reranking_result = await reranking_service.re_rank_results(
                        query=query,
                        results=combined_results,
                        top_k=rerank_top_k,
                        show_scores=show_scores,
                        show_justification=show_justification
                    )
                    
                    return {
                        "query": query,
                        "results": reranking_result["re_ranked_results"],
                        "dense_count": len(dense_results),
                        "sparse_count": len(sparse_results),
                        "total_results": reranking_result["total_results"],
                        "re_ranking": {
                            "enabled": True,
                            "successful": reranking_result["re_ranking_successful"],
                            "evaluated_count": reranking_result.get("evaluated_count", 0),
                            "error": reranking_result.get("error")
                        }
                    }
                    
                except Exception as e:
                    logging.error(f"Re-ranking failed, returning original results: {e}")
                    return {
                        "query": query,
                        "results": combined_results[:rerank_top_k],
                        "dense_count": len(dense_results),
                        "sparse_count": len(sparse_results),
                        "total_results": min(len(combined_results), rerank_top_k),
                        "re_ranking": {
                            "enabled": True,
                            "successful": False,
                            "error": str(e)
                        }
                    }
            else:
                # Return original results without re-ranking
                return {
                    "query": query,
                    "results": combined_results,
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "total_results": len(combined_results),
                    "re_ranking": {
                        "enabled": False,
                        "successful": None,
                        "error": None
                    }
                }
            
        except Exception as e:
            logging.error(f"Hybrid search failed: {e}")
            raise
    
    async def _search_dense(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search the dense collection."""
        # Check if collection exists
        if not utility.has_collection(self.dense_collection_name):
            logging.warning(f"Collection {self.dense_collection_name} does not exist.")
            return []
        
        collection = Collection(self.dense_collection_name)
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 8}},
            limit=top_k,
            output_fields=["chunk_id", "repo_name", "file_path", "text", "chunk_type", "file_language", "author"]
        )
        
        hits = results[0]
        return [
            {
                "chunk_id": hit.entity.get("chunk_id"),
                "repo_name": hit.entity.get("repo_name"),
                "file_path": hit.entity.get("file_path"),
                "text": hit.entity.get("text"),
                "chunk_type": hit.entity.get("chunk_type"),
                "file_language": hit.entity.get("file_language"),
                "author": hit.entity.get("author"),
                "score": hit.distance,
                "search_type": "dense"
            }
            for hit in hits
        ]
    
    async def _search_sparse(self, query_tfidf: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search the sparse collection."""
        # Check if collection exists
        if not utility.has_collection(self.sparse_collection_name):
            logging.warning(f"Collection {self.sparse_collection_name} does not exist.")
            return []
        
        collection = Collection(self.sparse_collection_name)
        
        results = collection.search(
            data=[query_tfidf.tolist()],
            anns_field="tfidf_vector",
            param={"metric_type": "IP", "params": {"nprobe": 8}},
            limit=top_k,
            output_fields=["chunk_id", "repo_name", "file_path", "text", "chunk_type", "file_language", "author"]
        )
        
        hits = results[0]
        return [
            {
                "chunk_id": hit.entity.get("chunk_id"),
                "repo_name": hit.entity.get("repo_name"),
                "file_path": hit.entity.get("file_path"),
                "text": hit.entity.get("text"),
                "chunk_type": hit.entity.get("chunk_type"),
                "file_language": hit.entity.get("file_language"),
                "author": hit.entity.get("author"),
                "score": hit.distance,
                "search_type": "sparse"
            }
            for hit in hits
        ]
    
    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict], top_k: int) -> List[Dict]:
        """Combine and rank results from both collections."""
        # Create a map of chunk_id to results
        result_map = {}
        
        # Add dense results
        for result in dense_results:
            chunk_id = result['chunk_id']
            if chunk_id not in result_map:
                result_map[chunk_id] = result
                result_map[chunk_id]['combined_score'] = result['score'] * 0.7  # Weight dense more
            else:
                # If already exists, combine scores
                result_map[chunk_id]['combined_score'] = (
                    result_map[chunk_id]['combined_score'] * 0.3 + result['score'] * 0.7
                )
        
        # Add sparse results
        for result in sparse_results:
            chunk_id = result['chunk_id']
            if chunk_id not in result_map:
                result_map[chunk_id] = result
                result_map[chunk_id]['combined_score'] = result['score'] * 0.3  # Weight sparse less
            else:
                # If already exists, combine scores
                result_map[chunk_id]['combined_score'] = (
                    result_map[chunk_id]['combined_score'] * 0.7 + result['score'] * 0.3
                )
        
        # Sort by combined score and return top_k
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results[:top_k]
    
    async def get_repository_status(self, repo_name: str) -> Dict[str, Any]:
        """Get processing status for a repository."""
        try:
            dense_count = 0
            sparse_count = 0
            
            # Check if dense collection exists
            if utility.has_collection(self.dense_collection_name):
                dense_collection = Collection(self.dense_collection_name)
                dense_count = len(dense_collection.query(
                    expr=f'repo_name == "{repo_name}"',
                    output_fields=["chunk_id"]
                ))
            
            # Check if sparse collection exists
            if utility.has_collection(self.sparse_collection_name):
                sparse_collection = Collection(self.sparse_collection_name)
                sparse_count = len(sparse_collection.query(
                    expr=f'repo_name == "{repo_name}"',
                    output_fields=["chunk_id"]
                ))
            
            return {
                "repo_name": repo_name,
                "dense_chunks": dense_count,
                "sparse_chunks": sparse_count,
                "total_chunks": dense_count + sparse_count
            }
            
        except Exception as e:
            logging.error(f"Failed to get repository status: {e}")
            return {
                "repo_name": repo_name,
                "dense_chunks": 0,
                "sparse_chunks": 0,
                "total_chunks": 0,
                "error": str(e)
            } 