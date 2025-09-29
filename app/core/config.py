from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Network Admin"
    DB_URL: str = "postgresql+psycopg2://user:password@db:5432/netadmin"
    REDIS_URL: str = "redis://redis:6379/0"
    GNN_MODEL_PATH: str = "./models/gnn_model.pt"

    class Config:
        env_file = ".env"

settings = Settings()
