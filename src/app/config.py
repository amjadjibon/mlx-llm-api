from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "MLX LLM API"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # MLX Settings
    llm_model_directory: str
    llm_model_name: str
    llm_model_max_tokens: int = 1000
    llm_model_temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 