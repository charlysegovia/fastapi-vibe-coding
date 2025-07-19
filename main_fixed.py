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

# Create FastAPI app first
app = FastAPI(
    title="GitHub RAG System",
    description="A RAG system that processes GitHub repositories with dual indexing (dense + sparse)",
    version="2.0.0"
)

# Global variable for load_data_service
load_data_service = None

# Initialize load_data_service on startup
@app.on_event("startup")
async def startup_event():
    global load_data_service
    try:
        from load_data import LoadDataService
        load_data_service = LoadDataService()
        await load_data_service.initialize()
        logging.info("LoadDataService initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize LoadDataService: {e}")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down application")

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