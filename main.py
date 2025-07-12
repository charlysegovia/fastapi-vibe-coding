import os
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

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
def lifespan(app: FastAPI):
    # Initialize Milvus and OpenAI clients here if needed
    # Store them in app.state
    yield
    # Release resources here if needed

app = FastAPI(lifespan=lifespan)

from routers.upload_routes import router as upload_router
from routers.query_routes import router as query_router

# Mount routers
app.include_router(upload_router)
app.include_router(query_router)

@app.get("/")
def root():
    """Root endpoint for health check."""
    return {"status": "ok"} 