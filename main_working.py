import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging
from fastapi.staticfiles import StaticFiles
import asyncio

# Load environment variables
load_dotenv()

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Create FastAPI app first
app = FastAPI(
    title="GitHub RAG System",
    description="A RAG system that processes GitHub repositories with dual indexing (dense + sparse)",
    version="2.0.0"
)

# Global variable for load_data_service
load_data_service = None
initialization_complete = False

async def initialize_services():
    """Initialize all services with proper error handling."""
    global load_data_service, initialization_complete
    
    try:
        logging.info("Starting service initialization...")
        
        # Import and initialize LoadDataService
        from load_data import LoadDataService
        load_data_service = LoadDataService()
        await load_data_service.initialize()
        
        # Test the service
        repos = await load_data_service.list_repositories()
        logging.info(f"Service initialization successful! Found {len(repos)} repositories")
        
        initialization_complete = True
        return True
        
    except Exception as e:
        logging.error(f"Service initialization failed: {e}")
        initialization_complete = True  # Mark as complete even if failed
        return False

def get_service_status():
    """Get current service status."""
    return {
        "initialization_complete": initialization_complete,
        "service_available": load_data_service is not None,
        "service_ready": initialization_complete and load_data_service is not None
    }

def get_load_data_service():
    """Get the load data service with proper error handling."""
    if load_data_service is None:
        raise RuntimeError("LoadDataService not initialized")
    return load_data_service

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logging.info("Application startup initiated")
    success = await initialize_services()
    if success:
        logging.info("Application startup completed successfully")
    else:
        logging.error("Application startup completed with errors")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down application")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    """Root endpoint for health check."""
    status = get_service_status()
    return {
        "status": "ok",
        "message": "GitHub RAG System is running",
        "version": "2.0.0",
        "service_status": status,
        "endpoints": {
            "github": "/api/repos",
            "query": "/api/query",
            "health": "/api/health",
            "init_status": "/api/init-status",
            "web_interface": "/static/index.html"
        }
    }

@app.get("/api/init-status")
async def get_init_status():
    """Get initialization status."""
    return get_service_status()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Import routers after app is created to avoid circular imports
def setup_routes():
    """Setup routes after app is created."""
    from routers.github_routes import router as github_router
    from routers.query_routes import router as query_router
    
    # Mount routers
    app.include_router(github_router, prefix="/api", tags=["GitHub"])
    app.include_router(query_router, prefix="/api", tags=["Query"])

# Setup routes
setup_routes() 