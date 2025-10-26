"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, logging
from contextlib import asynccontextmanager
from app.config import settings
from app.api.routes import inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Application starting...")
    yield
    logger.info("🛑 Application shutting down...")


# Create app
app = FastAPI(
    title="DDoS Detection Platform",
    description="Real-time DDoS detection using FTG-NET",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",          # Explicitly set Swagger UI path
    openapi_url="/openapi.json" # Explicitly set OpenAPI schema path
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(inference.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 DDoS Detection Platform",
        "version": "1.0.0",
        "model": "FTG-NET",
        "status": "✅ online",
        "docs": "Visit http://localhost:8000/docs for Swagger UI",
        "endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/api/inference/health",
            "stats": "/api/inference/stats",
            "predict": "/api/inference/predict-batch"
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors"""
    return FileResponse("favicon.ico", media_type="image/x-icon") if os.path.exists("favicon.ico") else {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port, 
        reload=settings.reload,
        log_level="info"
    )
