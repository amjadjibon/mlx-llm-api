from fastapi import APIRouter, HTTPException, Depends, status
import logging
from ...models import CompletionRequest, CompletionResponse, ErrorResponse, ErrorDetail
from ...services.mlx_service import MLXService
from ...core.dependencies import get_mlx_service_dependency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["OpenAI Completions"])


@router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Create a completion",
    description="Create a text completion using the loaded MLX model (OpenAI-compatible)",
    responses={
        200: {"description": "Successful completion", "model": CompletionResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Model not available", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_completion(
    request: CompletionRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> CompletionResponse:
    """Create a text completion using the loaded MLX model (OpenAI completions API)."""
    try:
        logger.info(f"Processing completion request for model: {request.model}")

        # Handle single prompt or list of prompts
        if isinstance(request.prompt, list):
            if len(request.prompt) > 1:
                # MLX typically handles one prompt at a time
                logger.warning("Multiple prompts requested, using first prompt only")
            prompt = request.prompt[0] if request.prompt else ""
        else:
            prompt = request.prompt

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="Prompt cannot be empty",
                    code="EMPTY_PROMPT",
                ).model_dump(),
            )

        # Check for unsupported parameters
        if request.n and request.n > 1:
            logger.warning(
                f"Multiple completions requested (n={request.n}), MLX supports only 1"
            )

        if request.stream:
            # TODO: Implement streaming
            logger.warning("Streaming requested but not yet implemented")

        response = await mlx_service.generate_text_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
            if isinstance(request.stop, list)
            else [request.stop]
            if request.stop
            else None,
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
