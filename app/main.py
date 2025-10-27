"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import logging
from contextlib import asynccontextmanager
from app.config import settings
from app.api.routes import inference, pcap

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
    docs_url="/docs",
    openapi_url="/openapi.json"
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
app.include_router(pcap.router)


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
            "predict": "/api/inference/predict",
            "evaluate": "/api/inference/evaluate",
            "pcap_convert": "/api/pcap/convert"
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors"""
    return FileResponse("favicon.ico", media_type="image/x-icon") if os.path.exists("favicon.ico") else {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )
