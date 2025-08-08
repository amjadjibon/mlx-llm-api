import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.dependencies import get_mlx_service_dependency
from ...models import EmbeddingRequest, EmbeddingResponse, ErrorDetail, ErrorResponse
from ...services.mlx_service import MLXService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["OpenAI Embeddings"])


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings",
    description="Create embeddings for input text(s) (OpenAI-compatible)",
    responses={
        200: {"description": "Successful embeddings", "model": EmbeddingResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Model not available", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_embeddings(
    request: EmbeddingRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> EmbeddingResponse:
    """Create embeddings for input text(s) using available models (OpenAI embeddings API)."""
    try:
        logger.info(f"Processing embeddings request for model: {request.model}")

        # Handle single input or list of inputs
        if isinstance(request.input, list):
            input_texts = request.input
        else:
            input_texts = [request.input]

        if not input_texts or not any(text.strip() for text in input_texts):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="Input cannot be empty",
                    code="EMPTY_INPUT",
                ).model_dump(),
            )

        # Check if the requested model appears to be an embedding model
        if not mlx_service.is_embedding_model(request.model):
            logger.warning(
                f"Model '{request.model}' may not be optimized for embeddings"
            )

        response = await mlx_service.generate_embeddings(
            input_texts=input_texts, model=request.model
        )

        logger.info(f"Successfully generated embeddings for {len(input_texts)} inputs")
        return response

    except RuntimeError as e:
        logger.error(f"Runtime error during embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                type="model_error", message=str(e), code="MODEL_RUNTIME_ERROR"
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected error during embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                type="internal_error",
                message="An unexpected error occurred while generating embeddings",
                code="EMBEDDINGS_ERROR",
            ).model_dump(),
        )
