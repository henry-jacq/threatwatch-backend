"""
Application configuration
"""
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "DDoS Detection Platform"
    debug: bool = True
    api_version: str = "v1"
    
    # Model
    model_checkpoint: str = "models/checkpoints_v4_metadata/best_model_1.pt"
    device: str = "auto"  # auto, cuda, cpu
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    class Config:
        env_file = ".env"
        protected_namespaces = ('settings_',)  # Fix Pydantic warning


settings = Settings()
