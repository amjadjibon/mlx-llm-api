from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Settings
    app_name: str = Field(default="MLX LLM API", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    
    # Security Settings
    allowed_hosts: list[str] = Field(default=["*"], description="Allowed hosts")
    cors_origins: list[str] = Field(default=["*"], description="CORS origins")
    
    # MLX Settings
    llm_model_directory: Optional[str] = Field(default=None, description="Path to MLX model directory")
    llm_model_name: Optional[str] = Field(default=None, description="MLX model name")
    llm_model_max_tokens: int = Field(default=1000, ge=1, le=8192, description="Default max tokens")
    llm_model_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default temperature")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance for backward compatibility
settings = get_settings()
