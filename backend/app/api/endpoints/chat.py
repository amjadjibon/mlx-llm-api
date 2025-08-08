import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.dependencies import get_mlx_service_dependency
from ...models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorDetail,
    ErrorResponse,
)
from ...services.mlx_service import MLXService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/chat", tags=["OpenAI Chat Completions"])


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    summary="Create a chat completion",
    description="Create a chat completion using the loaded MLX model",
    responses={
        200: {"description": "Successful completion", "model": ChatCompletionResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Model not loaded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> ChatCompletionResponse:
    """Create a chat completion using the loaded MLX model (OpenAI-compatible)."""
    try:
        logger.info(f"Processing chat completion request for model: {request.model}")

        # Check for unsupported parameters and log warnings
        if request.n and request.n > 1:
            logger.warning(
                f"Multiple completions requested (n={request.n}), MLX supports only 1"
            )

        if request.tools:
            logger.warning("Tools/function calling requested but not supported")

        if request.response_format:
            logger.info("Response format specified - basic JSON mode may be supported")

        if request.stream:
            logger.warning("Streaming requested but using non-streaming endpoint")

        response = await mlx_service.generate_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model=request.model,
        )
        logger.info(f"Successfully generated completion with ID: {response.id}")
        return response
    except RuntimeError as e:
        logger.error(f"Runtime error during completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                type="model_error", message=str(e), code="MODEL_RUNTIME_ERROR"
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected error during completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                type="internal_error",
                message="An unexpected error occurred while generating completion",
                code="COMPLETION_ERROR",
            ).model_dump(),
        )


@router.post(
    "/completions/stream",
    summary="Create a streaming chat completion",
    description="Create a streaming chat completion (not yet implemented)",
    responses={
        501: {"description": "Not implemented", "model": ErrorResponse},
        503: {"description": "Model not loaded", "model": ErrorResponse},
    },
)
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
):
    """Create a streaming chat completion (not yet implemented)."""
    del request, mlx_service  # Avoid unused variable warnings
    logger.warning("Streaming completion requested but not yet implemented")
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=ErrorDetail(
            type="not_implemented",
            message="Streaming completions are not yet implemented",
            code="STREAMING_NOT_IMPLEMENTED",
        ).model_dump(),
    )
