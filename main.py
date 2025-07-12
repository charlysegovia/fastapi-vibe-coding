import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager
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

# Lifespan context manager for global resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Milvus and OpenAI clients here if needed
    # Store them in app.state
    yield
    # Release resources here if needed

app = FastAPI(lifespan=lifespan)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return JSON error responses."""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__
        }
    )

from routers.upload_routes import router as upload_router
from routers.query_routes import router as query_router

# Mount routers
app.include_router(upload_router)
app.include_router(query_router)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    """Root endpoint for health check."""
    return {"status": "ok"} 