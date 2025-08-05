from fastapi import APIRouter, HTTPException, Depends, status
import time
import logging
from ...models import (
    ModelListResponse,
    ModelInfo,
    ErrorResponse,
    ErrorDetail
)
from ...services.mlx_service import MLXService
from ...core.dependencies import get_mlx_service_dependency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["OpenAI Models"])


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List models",
    description="List available models (OpenAI-compatible)",
    responses={
        200: {"description": "List of models", "model": ModelListResponse}
    }
)
async def list_models(
    mlx_service: MLXService = Depends(get_mlx_service_dependency)
) -> ModelListResponse:
    """List all available models (OpenAI-compatible /v1/models endpoint)."""
    models = []
    available_models = mlx_service.get_available_models()
    current_time = int(time.time())
    
    if available_models:
        for model_name in available_models.keys():
            models.append(
                ModelInfo(
                    id=model_name,
                    object="model",
                    created=current_time,
                    owned_by="mlx-llm-api"
                )
            )
        logger.info(f"Listed {len(models)} available models via /v1/models")
    else:
        logger.warning("No models available in model directory")
    
    return ModelListResponse(
        object="list",
        data=models
    )


@router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    summary="Retrieve model",
    description="Get details about a specific model (OpenAI-compatible)",
    responses={
        200: {"description": "Model details", "model": ModelInfo},
        404: {"description": "Model not found", "model": ErrorResponse}
    }
)
async def get_model(
    model_id: str,
    mlx_service: MLXService = Depends(get_mlx_service_dependency)
) -> ModelInfo:
    """Get details about a specific model (OpenAI-compatible /v1/models/{id} endpoint)."""
    available_models = mlx_service.get_available_models()
    
    if model_id not in available_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorDetail(
                type="model_not_found",
                message=f"Model '{model_id}' not found",
                code="MODEL_NOT_FOUND",
                details={"available_models": list(available_models.keys())}
            ).model_dump()
        )
    
    logger.info(f"Retrieved model details for: {model_id}")
    return ModelInfo(
        id=model_id,
        object="model",
        created=int(time.time()),
        owned_by="mlx-llm-api"
    )