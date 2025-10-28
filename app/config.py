from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int = 5432

    # Ollama
    OLLAMA_HOST: str
    OLLAMA_RERANK_HOST: str
    EMBEDDING_MODEL: str = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
    RERANK_MODEL: str = "gemma3:27b"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
