from mlx_lm import load, generate
from typing import List, Optional
import time
import uuid
from ..models import ChatMessage, ChatCompletionResponse
from ..config import settings


class MLXService:
    """Service for handling MLX model operations."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
    async def load_model(self, model_path: str) -> bool:
        """Load the MLX model."""
        try:
            self.model, self.tokenizer = load(model_path)
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for the model."""
        formatted = ""
        for message in messages:
            if message.role == "system":
                formatted += f"System: {message.content}\n"
            elif message.role == "user":
                formatted += f"User: {message.content}\n"
            elif message.role == "assistant":
                formatted += f"Assistant: {message.content}\n"
        formatted += "Assistant: "
        return formatted
    
    async def generate_completion(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Format messages
        prompt = self._format_messages(messages)
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        
        # Generate
        start_time = time.time()
        response_tokens = generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens or settings.llm_model_max_tokens,
            temperature=temperature or settings.llm_model_temperature
        )
        end_time = time.time()
        
        # Decode response
        response_text = self.tokenizer.decode(response_tokens)
        
        # Calculate usage
        prompt_tokens = len(tokens)
        completion_tokens = len(response_tokens)
        total_tokens = prompt_tokens + completion_tokens
        
        # Create response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model="mlx-model",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loaded


# Global service instance
mlx_service = MLXService()
