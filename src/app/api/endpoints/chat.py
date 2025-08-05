from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ...models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse
)
from ...services.mlx_service import mlx_service
from ...config import settings

router = APIRouter(prefix="/v1/chat", tags=["chat"])


async def verify_model_loaded():
    """Dependency to verify model is loaded."""
    if not mlx_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load a model first."
        )


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    responses={
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse}
    }
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    _: bool = Depends(verify_model_loaded)
) -> ChatCompletionResponse:
    """Create a chat completion."""
    try:
        response = await mlx_service.generate_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating completion: {str(e)}"
        )


@router.post(
    "/completions/stream",
    responses={
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse}
    }
)
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    _: bool = Depends(verify_model_loaded)
):
    """Create a streaming chat completion."""
    # TODO: Implement streaming response
    raise HTTPException(
        status_code=501,
        detail="Streaming not yet implemented"
    ) 