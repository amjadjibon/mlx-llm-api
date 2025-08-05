from fastapi import APIRouter, HTTPException, Depends, status
import time
import logging
from ...models import (
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
    LoadModelRequest,
    LoadModelResponse,
    ModelListResponse,
    ModelInfo
)
from ...services.mlx_service import MLXService
from ...core.dependencies import get_mlx_service_dependency, get_current_settings
from ...config import Settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health & Management"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and model",
    responses={
        200: {"description": "Service is healthy", "model": HealthResponse}
    }
)
async def health_check(
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
    settings: Settings = Depends(get_current_settings)
) -> HealthResponse:
    """Health check endpoint."""
    model_loaded = mlx_service.is_model_loaded()
    model_info = mlx_service.get_model_info() if model_loaded else None
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=model_loaded,
        model_info=model_info
    )


@router.post(
    "/load-model",
    response_model=LoadModelResponse,
    summary="Load a model",
    description="Load an MLX model from the specified path",
    responses={
        200: {"description": "Model loaded successfully", "model": LoadModelResponse},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {"description": "Model loading failed", "model": ErrorResponse}
    }
)
async def load_model(
    request: LoadModelRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency)
) -> LoadModelResponse:
    """Load a model from the specified path."""
    try:
        logger.info(f"Loading model from: {request.model_path}")
        success = await mlx_service.load_model(
            model_path=request.model_path,
            model_name=request.model_name
        )
        
        if success:
            model_name = request.model_name or "mlx-model"
            logger.info(f"Successfully loaded model: {model_name}")
            return LoadModelResponse(
                success=True,
                message=f"Model loaded successfully from {request.model_path}",
                model_name=model_name
            )
        else:
            logger.error(f"Failed to load model from: {request.model_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorDetail(
                    type="model_load_error",
                    message="Failed to load model",
                    code="MODEL_LOAD_FAILED"
                ).model_dump()
            )
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                type="invalid_request",
                message=f"Error loading model: {str(e)}",
                code="MODEL_LOAD_ERROR"
            ).model_dump()
        )


@router.post(
    "/unload-model",
    summary="Unload the current model",
    description="Unload the currently loaded model to free memory",
    responses={
        200: {"description": "Model unloaded successfully"},
        400: {"description": "No model loaded", "model": ErrorResponse}
    }
)
async def unload_model(
    mlx_service: MLXService = Depends(get_mlx_service_dependency)
):
    """Unload the current model."""
    if not mlx_service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                type="no_model_loaded",
                message="No model is currently loaded",
                code="NO_MODEL_LOADED"
            ).model_dump()
        )
    
    mlx_service.unload_model()
    logger.info("Model unloaded successfully")
    return {"message": "Model unloaded successfully"}


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="List all available models",
    responses={
        200: {"description": "List of models", "model": ModelListResponse}
    }
)
async def list_models(
    mlx_service: MLXService = Depends(get_mlx_service_dependency)
) -> ModelListResponse:
    """List available models."""
    models = []
    
    if mlx_service.is_model_loaded():
        model_info = mlx_service.get_model_info()
        models.append(
            ModelInfo(
                id=model_info["model_name"],
                object="model",
                created=int(time.time()),
                owned_by="mlx-llm-api"
            )
        )
    else:
        # Default model entry when no model is loaded
        models.append(
            ModelInfo(
                id="mlx-model",
                object="model",
                created=int(time.time()),
                owned_by="mlx-llm-api"
            )
        )
    
    return ModelListResponse(
        object="list",
        data=models
    )
