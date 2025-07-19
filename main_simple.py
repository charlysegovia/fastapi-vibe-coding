import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Create FastAPI app
app = FastAPI(
    title="GitHub RAG System",
    description="A RAG system that processes GitHub repositories with dual indexing (dense + sparse)",
    version="2.0.0"
)

# Import routers
from routers.github_routes import router as github_router
from routers.query_routes import router as query_router

# Mount routers
app.include_router(github_router, prefix="/api", tags=["GitHub"])
app.include_router(query_router, prefix="/api", tags=["Query"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    """Root endpoint for health check."""
    return {
        "status": "ok",
        "message": "GitHub RAG System is running",
        "version": "2.0.0",
        "endpoints": {
            "github": "/api/repos",
            "query": "/api/query",
            "health": "/api/health",
            "web_interface": "/static/index.html"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Initialize services lazily when first needed
_load_data_service = None

def get_load_data_service():
    """Get or create the load_data_service instance."""
    global _load_data_service
    if _load_data_service is None:
        try:
            from load_data import LoadDataService
            _load_data_service = LoadDataService()
            logging.info("LoadDataService created (will initialize on first use)")
        except Exception as e:
            logging.error(f"Failed to create LoadDataService: {e}")
            raise
    return _load_data_service

# Update the routers to use lazy initialization
@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint to verify GitHub and Milvus connectivity."""
    try:
        service = get_load_data_service()
        # Test GitHub connectivity
        repos = await service.list_repositories()
        
        # Test Milvus connectivity by getting status of a repo (if any exist)
        milvus_status = "unknown"
        if repos:
            try:
                status = await service.get_processing_status(repos[0]['name'])
                milvus_status = "connected" if status.get('total_chunks', 0) >= 0 else "error"
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