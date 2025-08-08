import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from ...config import Settings
from ...core.dependencies import get_current_settings, get_mlx_service_dependency
from ...models import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    ModelListResponse,
)
from ...services.mlx_service import MLXService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health & Management"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and model",
    responses={200: {"description": "Service is healthy", "model": HealthResponse}},
)
async def health_check(
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
    settings: Settings = Depends(get_current_settings),
) -> HealthResponse:
    """Health check endpoint."""
    model_loaded = mlx_service.is_model_loaded()
    model_info = mlx_service.get_model_info() if model_loaded else None

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=model_loaded,
        model_info=model_info,
    )


@router.post(
    "/load-model",
    response_model=LoadModelResponse,
    summary="Load a model",
    description="Load an MLX model from the specified path",
    responses={
        200: {"description": "Model loaded successfully", "model": LoadModelResponse},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {"description": "Model loading failed", "model": ErrorResponse},
    },
)
async def load_model(
    request: LoadModelRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> LoadModelResponse:
    """Load a model from the specified path."""
    try:
        logger.info(f"Loading model from: {request.model_path}")
        success = await mlx_service.load_model(
            model_path=request.model_path, model_name=request.model_name
        )

        if success:
            model_name = request.model_name or "mlx-model"
            logger.info(f"Successfully loaded model: {model_name}")
            return LoadModelResponse(
                success=True,
                message=f"Model loaded successfully from {request.model_path}",
                model_name=model_name,
            )
        else:
            logger.error(f"Failed to load model from: {request.model_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorDetail(
                    type="model_load_error",
                    message="Failed to load model",
                    code="MODEL_LOAD_FAILED",
                ).model_dump(),
            )
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                type="invalid_request",
                message=f"Error loading model: {str(e)}",
                code="MODEL_LOAD_ERROR",
            ).model_dump(),
        )


@router.post(
    "/unload-model",
    summary="Unload the current model",
    description="Unload the currently loaded model to free memory",
    responses={
        200: {"description": "Model unloaded successfully"},
        400: {"description": "No model loaded", "model": ErrorResponse},
    },
)
async def unload_model(mlx_service: MLXService = Depends(get_mlx_service_dependency)):
    """Unload the current model."""
    if not mlx_service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                type="no_model_loaded",
                message="No model is currently loaded",
                code="NO_MODEL_LOADED",
            ).model_dump(),
        )

    mlx_service.unload_model()
    logger.info("Model unloaded successfully")
    return {"message": "Model unloaded successfully"}


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="List all available models in the model directory (OpenAI-compatible)",
    responses={200: {"description": "List of models", "model": ModelListResponse}},
)
async def list_models(
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> ModelListResponse:
    """List all available models (OpenAI-compatible endpoint)."""
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
                    owned_by="mlx-llm-api",
                )
            )
        logger.info(f"Listed {len(models)} available models")
    else:
        logger.warning("No models available in model directory")

    return ModelListResponse(object="list", data=models)


@router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    summary="Retrieve a model",
    description="Get details about a specific model (OpenAI-compatible)",
    responses={
        200: {"description": "Model details", "model": ModelInfo},
        404: {"description": "Model not found", "model": ErrorResponse},
    },
)
async def get_model(
    model_id: str, mlx_service: MLXService = Depends(get_mlx_service_dependency)
) -> ModelInfo:
    """Get details about a specific model (OpenAI-compatible endpoint)."""
    available_models = mlx_service.get_available_models()

    if model_id not in available_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorDetail(
                type="model_not_found",
                message=f"Model '{model_id}' not found",
                code="MODEL_NOT_FOUND",
                details={"available_models": list(available_models.keys())},
            ).model_dump(),
        )

    return ModelInfo(
        id=model_id, object="model", created=int(time.time()), owned_by="mlx-llm-api"
    )


@router.post(
    "/switch-model",
    summary="Switch to a different model",
    description="Switch to a different available model",
    responses={
        200: {"description": "Model switched successfully"},
        400: {
            "description": "Invalid model or model not available",
            "model": ErrorResponse,
        },
        500: {"description": "Failed to switch model", "model": ErrorResponse},
    },
)
async def switch_model(
    model_name: str, mlx_service: MLXService = Depends(get_mlx_service_dependency)
):
    """Switch to a different model."""
    available_models = mlx_service.get_available_models()

    if model_name not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                type="invalid_model",
                message=f"Model '{model_name}' not found",
                code="MODEL_NOT_FOUND",
                details={"available_models": list(available_models.keys())},
            ).model_dump(),
        )

    try:
        logger.info(f"Switching to model: {model_name}")
        success = await mlx_service.ensure_model_loaded(model_name)

        if success:
            logger.info(f"Successfully switched to model: {model_name}")
            return {
                "message": f"Successfully switched to model: {model_name}",
                "current_model": model_name,
                "available_models": list(available_models.keys()),
            }
        else:
            logger.error(f"Failed to switch to model: {model_name}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorDetail(
                    type="model_switch_error",
                    message=f"Failed to switch to model: {model_name}",
                    code="MODEL_SWITCH_FAILED",
                ).model_dump(),
            )
    except Exception as e:
        logger.error(f"Error switching to model {model_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                type="model_switch_error",
                message=f"Error switching to model: {str(e)}",
                code="MODEL_SWITCH_ERROR",
            ).model_dump(),
        )
