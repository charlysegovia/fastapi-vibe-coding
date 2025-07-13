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
    # Here Milvus and OpenAI clients will be initialized
    # and stored in app.state
    yield
    # Here resources will be released if needed

app = FastAPI(lifespan=lifespan)

# Import and mount routers (to be implemented)
# from routers.upload_routes import router as upload_router
# from routers.query_routes import router as query_router
# app.include_router(upload_router)
# app.include_router(query_router)

@app.get("/")
def root():
    """Root endpoint for health check."""
    return {"status": "ok"} 