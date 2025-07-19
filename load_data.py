import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import mimetypes
from pathlib import Path

# Import our modules
from github_service import GitHubService
from file_processor import FileProcessor
from chunking_strategies import ChunkingStrategies
from milvus_dual import MilvusDualService

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB default
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
    '.md', '.txt', '.rst', '.tex', '.adoc', '.wiki',
    '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.sql', '.sh', '.bat', '.ps1', '.dockerfile', '.dockerignore',
    '.gitignore', '.gitattributes', '.editorconfig', '.eslintrc', '.prettierrc'
}

class LoadDataService:
    """Main service for loading GitHub repositories to Milvus with dual indexing."""
    
    def __init__(self):
        self.github_service = GitHubService()
        self.file_processor = FileProcessor()
        self.chunking_strategies = ChunkingStrategies()
        self.milvus_service = MilvusDualService()
        
    async def initialize(self):
        """Initialize all services."""
        await self.milvus_service.initialize()
        
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all repositories for the authenticated user."""
        return await self.github_service.list_repositories()
        
    async def get_repository_files(self, repo_name: str) -> List[Dict[str, Any]]:
        """Get all text files from a repository."""
        files = await self.github_service.get_repository_files(repo_name)
        return self.file_processor.filter_text_files(files)
        
    async def process_repository(self, repo_name: str, progress_callback=None) -> Dict[str, Any]:
        """Process a complete repository and load it to Milvus."""
        logging.info(f"Starting processing of repository: {repo_name}")
        
        # Get repository info
        repo_info = await self.github_service.get_repository_info(repo_name)
        
        # Get all text files
        files = await self.get_repository_files(repo_name)
        logging.info(f"Found {len(files)} text files to process")
        
        if progress_callback:
            progress_callback(0, f"Found {len(files)} files to process")
        
        total_chunks = 0
        processed_files = 0
        errors = []
        dense_chunks_total = 0
        sparse_chunks_total = 0
        
        for i, file_info in enumerate(files):
            try:
                # Download file content
                content = await self.github_service.get_file_content(repo_name, file_info['path'])
                
                # Validate file
                if not self.file_processor.is_valid_text_file(content, file_info['size']):
                    logging.info(f"Skipping invalid file: {file_info['path']}")
                    continue
                
                # Create chunks
                dense_chunks = self.chunking_strategies.create_dense_chunks(
                    content, file_info, repo_info
                )
                sparse_chunks = self.chunking_strategies.create_sparse_chunks(
                    content, file_info, repo_info
                )
                
                # Only count as processed if we have chunks
                if dense_chunks or sparse_chunks:
                    # Insert into Milvus
                    if dense_chunks:
                        try:
                            await self.milvus_service.insert_dense_chunks(dense_chunks)
                            dense_chunks_total += len(dense_chunks)
                        except Exception as e:
                            logging.error(f"Failed to insert dense chunks for {file_info['path']}: {e}")
                            errors.append(f"Failed to insert dense chunks for {file_info['path']}: {str(e)}")
                    
                    if sparse_chunks:
                        try:
                            await self.milvus_service.insert_sparse_chunks(sparse_chunks)
                            sparse_chunks_total += len(sparse_chunks)
                        except Exception as e:
                            logging.error(f"Failed to insert sparse chunks for {file_info['path']}: {e}")
                            errors.append(f"Failed to insert sparse chunks for {file_info['path']}: {str(e)}")
                    
                    total_chunks += len(dense_chunks) + len(sparse_chunks)
                    processed_files += 1
                    
                    if progress_callback:
                        progress = int((i + 1) / len(files) * 100)
                        progress_callback(progress, f"Processed {file_info['path']} ({len(dense_chunks)} dense, {len(sparse_chunks)} sparse chunks)")
                else:
                    logging.info(f"No chunks created for {file_info['path']} - file may be too small or empty")
                    
            except Exception as e:
                error_msg = f"Error processing {file_info['path']}: {str(e)}"
                logging.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "repo_name": repo_name,
            "total_files": len(files),
            "processed_files": processed_files,
            "total_chunks": total_chunks,
            "dense_chunks": dense_chunks_total,
            "sparse_chunks": sparse_chunks_total,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Repository processing complete: {result}")
        return result
        
    async def search_hybrid(self, query: str, top_k: int = 10, enable_reranking: bool = True, 
                          rerank_top_k: int = 5, show_scores: bool = True, 
                          show_justification: bool = True) -> Dict[str, Any]:
        """Perform hybrid search using both dense and sparse collections with optional re-ranking."""
        return await self.milvus_service.search_hybrid(
            query=query, 
            top_k=top_k,
            enable_reranking=enable_reranking,
            rerank_top_k=rerank_top_k,
            show_scores=show_scores,
            show_justification=show_justification
        )
        
    async def get_processing_status(self, repo_name: str) -> Dict[str, Any]:
        """Get processing status for a repository."""
        return await self.milvus_service.get_repository_status(repo_name)

# Global instance
load_data_service = LoadDataService()

async def main():
    """Main function for testing."""
    await load_data_service.initialize()
    
    # List repositories
    repos = await load_data_service.list_repositories()
    print(f"Found {len(repos)} repositories")
    
    if repos:
        # Process first repository as example
        repo_name = repos[0]['name']
        print(f"Processing repository: {repo_name}")
        
        def progress_callback(progress, message):
            print(f"Progress: {progress}% - {message}")
        
        result = await load_data_service.process_repository(repo_name, progress_callback)
        print(f"Processing result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 