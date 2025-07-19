from fastapi import APIRouter, HTTPException, status, Request
from typing import List, Dict, Any
import logging
from main_working import get_load_data_service, get_service_status

router = APIRouter()

@router.get("/repos")
async def list_repositories(request: Request) -> List[Dict[str, Any]]:
    """
    List all repositories for the authenticated GitHub user.
    Returns a list of repository information including name, description, language, etc.
    """
    try:
        status = get_service_status()
        if not status["service_ready"]:
            raise HTTPException(status_code=503, detail=f"Service not ready. Status: {status}")
        
        service = get_load_data_service()
        repos = await service.list_repositories()
        logging.info(f"Retrieved {len(repos)} repositories")
        return repos
    except Exception as e:
        logging.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list repositories: {str(e)}")

@router.get("/repo/{repo_name}/files")
async def get_repository_files(request: Request, repo_name: str) -> List[Dict[str, Any]]:
    """
    Get all text files from a specific repository.
    Returns a list of file information including path, size, and metadata.
    """
    try:
        status = get_service_status()
        if not status["service_ready"]:
            raise HTTPException(status_code=503, detail=f"Service not ready. Status: {status}")
        
        service = get_load_data_service()
        files = await service.get_repository_files(repo_name)
        logging.info(f"Retrieved {len(files)} files from repository {repo_name}")
        return files
    except Exception as e:
        logging.error(f"Failed to get files for repository {repo_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get repository files: {str(e)}")

@router.post("/process/{repo_name}")
async def process_repository(request: Request, repo_name: str) -> Dict[str, Any]:
    """
    Process a complete repository and load it to Milvus with dual indexing.
    This will:
    1. Download all text files from the repository
    2. Create dense chunks (paragraphs and sentences)
    3. Create sparse chunks (files and sections)
    4. Generate embeddings and store in Milvus
    """
    try:
        logging.info(f"Starting processing of repository: {repo_name}")
        
        status = get_service_status()
        if not status["service_ready"]:
            raise HTTPException(status_code=503, detail=f"Service not ready. Status: {status}")
        
        # Simple progress callback for logging
        def progress_callback(progress: int, message: str):
            logging.info(f"Progress {progress}%: {message}")
        
        service = get_load_data_service()
        result = await service.process_repository(repo_name, progress_callback)
        logging.info(f"Repository processing completed: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Failed to process repository {repo_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process repository: {str(e)}")

@router.get("/process/{repo_name}/status")
async def get_processing_status(request: Request, repo_name: str) -> Dict[str, Any]:
    """
    Get the processing status for a repository.
    Returns information about how many chunks are stored in Milvus for this repository.
    """
    try:
        status = get_service_status()
        if not status["service_ready"]:
            raise HTTPException(status_code=503, detail=f"Service not ready. Status: {status}")
        
        service = get_load_data_service()
        repo_status = await service.get_processing_status(repo_name)
        return repo_status
    except Exception as e:
        logging.error(f"Failed to get status for repository {repo_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get repository status: {str(e)}")

@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Health check endpoint to verify GitHub and Milvus connectivity.
    """
    try:
        status = get_service_status()
        if not status["service_ready"]:
            return {
                "status": "unhealthy",
                "error": f"Service not ready. Status: {status}"
            }
        
        # Additional check for load_data_service
        try:
            service = get_load_data_service()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"LoadDataService not available: {str(e)}"
            }
        
        # Test GitHub connectivity
        repos = await service.list_repositories()
        
        # Test Milvus connectivity by getting status of a repo (if any exist)
        milvus_status = "unknown"
        if repos:
            try:
                repo_status = await service.get_processing_status(repos[0]['name'])
                milvus_status = "connected" if repo_status.get('total_chunks', 0) >= 0 else "error"
            except:
                milvus_status = "error"
        
        return {
            "status": "healthy",
            "github": "connected" if repos else "error",
            "milvus": milvus_status,
            "repositories_count": len(repos)
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        } 