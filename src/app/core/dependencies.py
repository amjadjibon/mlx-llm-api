from fastapi import Depends, HTTPException, status
from ..config import Settings, get_settings
from ..services.mlx_service import MLXService, get_mlx_service
from ..models import ErrorDetail


def get_current_settings() -> Settings:
    """Dependency to get current settings."""
    return get_settings()


def get_mlx_service_dependency() -> MLXService:
    """Dependency to get MLX service."""
    return get_mlx_service()


async def verify_model_loaded(
    mlx_service: MLXService = Depends(get_mlx_service_dependency)
) -> MLXService:
    """Dependency to verify model is loaded."""
    if not mlx_service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                type="model_not_loaded",
                message="Model not loaded. Please load a model first.",
                code="MODEL_NOT_LOADED"
            ).model_dump()
        )
    return mlx_service


async def get_model_required(
    mlx_service: MLXService = Depends(verify_model_loaded)
) -> MLXService:
    """Alias for verify_model_loaded for cleaner endpoint definitions."""
    return mlx_service
