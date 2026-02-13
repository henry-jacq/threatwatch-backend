"""
Application configuration
"""
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "Threatwatch"
    debug: bool = True
    api_version: str = "v1"
    
    # Model
    default_model_id: str = "ftg_net_v1"
    device: str = "auto"  # auto, cuda, cpu
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    redis_host: str = "localhost"
    
    class Config:
        env_file = ".env"
        protected_namespaces = ('settings_',)  # Fix Pydantic warning


settings = Settings()
