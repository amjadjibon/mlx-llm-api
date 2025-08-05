from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="List of completion choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any] = Field(..., description="Error details") 