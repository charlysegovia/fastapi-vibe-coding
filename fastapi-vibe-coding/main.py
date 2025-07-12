import os
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

# Cargar variables de entorno
load_dotenv()

# Configuración de logging a archivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Lifespan context manager para recursos globales
@asynccontextmanager
def lifespan(app: FastAPI):
    # Aquí se inicializarán los clientes de Milvus y OpenAI
    # y se almacenarán en app.state
    yield
    # Aquí se liberarán recursos si es necesario

app = FastAPI(lifespan=lifespan)

# Importar y montar routers (a implementar)
# from routers.upload_routes import router as upload_router
# from routers.query_routes import router as query_router
# app.include_router(upload_router)
# app.include_router(query_router)

@app.get("/")
def root():
    """Root endpoint for health check."""
    return {"status": "ok"} 