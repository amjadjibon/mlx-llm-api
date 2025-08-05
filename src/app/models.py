from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class MessageRole(str, Enum):
    """Valid message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(str, Enum):
    """Valid finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


class ChatMessage(BaseModel):
    """Chat message model."""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message", min_length=1)
    
    class Config:
        use_enum_values = True


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt", ge=0)
    completion_tokens: int = Field(..., description="Number of tokens in the completion", ge=0)
    total_tokens: int = Field(..., description="Total number of tokens used", ge=0)


class ChatCompletionChoice(BaseModel):
    """A completion choice."""
    index: int = Field(..., description="Choice index", ge=0)
    message: ChatMessage = Field(..., description="The completion message")
    finish_reason: FinishReason = Field(..., description="Reason for finishing")
    
    class Config:
        use_enum_values = True


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: str = Field(..., description="Model to use for completion", min_length=1)
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation", min_items=1)
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate", ge=1, le=8192)
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences", max_items=4)


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: Literal["chat.completion"] = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionChoice] = Field(..., description="List of completion choices", min_items=1)
    usage: ChatCompletionUsage = Field(..., description="Token usage statistics")


class ModelInfo(BaseModel):
    """Model information."""
    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = Field("model", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    owned_by: str = Field(..., description="Organization that owns the model")


class ModelListResponse(BaseModel):
    """List of available models."""
    object: Literal["list"] = Field("list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")


class LoadModelRequest(BaseModel):
    """Model loading request."""
    model_path: str = Field(..., description="Path to the model", min_length=1)
    model_name: Optional[str] = Field(None, description="Optional model name")


class LoadModelResponse(BaseModel):
    """Model loading response."""
    success: bool = Field(..., description="Whether loading was successful")
    message: str = Field(..., description="Status message")
    model_name: Optional[str] = Field(None, description="Loaded model name")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")


class ErrorDetail(BaseModel):
    """Error detail structure."""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: ErrorDetail = Field(..., description="Error details")
