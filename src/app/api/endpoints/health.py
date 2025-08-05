from fastapi import APIRouter, HTTPException
from ...models import HealthResponse, ErrorResponse
from ...services.mlx_service import mlx_service
from ...config import settings

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}}
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=mlx_service.is_model_loaded()
    )


@router.post(
    "/load-model",
    responses={
        200: {"description": "Model loaded successfully"},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def load_model(model_path: str):
    """Load a model."""
    try:
        success = await mlx_service.load_model(model_path)
        if success:
            return {"message": f"Model loaded successfully from {model_path}"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error loading model: {str(e)}"
        )


@router.get("/models")
async def list_models():
    """List available models."""
    # TODO: Implement model listing
    return {
        "data": [
            {
                "id": "mlx-model",
                "object": "model",
                "created": 0,
                "owned_by": "mlx-llm-api"
            }
        ],
        "object": "list"
    }
