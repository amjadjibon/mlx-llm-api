from mlx_lm import load, generate
from typing import List, Optional, Dict, Any
import time
import uuid
import logging
from functools import lru_cache
from ..models import ChatMessage, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage
from ..config import get_settings

logger = logging.getLogger(__name__)


class MLXService:
    """Service for handling MLX model operations."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_path: Optional[str] = None
        self.model_name: str = "mlx-model"
        
    async def load_model(self, model_path: str, model_name: Optional[str] = None) -> bool:
        """Load the MLX model."""
        try:
            logger.info(f"Loading MLX model from: {model_path}")
            self.model, self.tokenizer = load(model_path)
            self.model_loaded = True
            self.model_path = model_path
            if model_name:
                self.model_name = model_name
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            self.model_loaded = False
            self.model_path = None
            return False
    
    def unload_model(self) -> None:
        """Unload the current model."""
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_path = None
        logger.info("Model unloaded")
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for the model using tokenizer's chat template if available."""
        # Try to use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Convert messages to dict format for chat template
                message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
                prompt = self.tokenizer.apply_chat_template(
                    message_dicts, 
                    add_generation_prompt=True,
                    tokenize=False
                )
                logger.debug("Using tokenizer's chat template")
                return prompt
            except Exception as e:
                logger.warning(f"Failed to use chat template, falling back to manual formatting: {e}")
        
        # Fallback to manual formatting
        formatted_parts = []
        
        for message in messages:
            role = message.role.lower()
            if role == "system":
                formatted_parts.append(f"System: {message.content}")
            elif role == "user":
                formatted_parts.append(f"User: {message.content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {message.content}")
            else:
                logger.warning(f"Unknown message role: {role}")
                formatted_parts.append(f"{role.capitalize()}: {message.content}")
        
        formatted_parts.append("Assistant:")
        return "\n".join(formatted_parts)
    
    async def generate_completion(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        settings = get_settings()
        
        # Use provided parameters or fallback to settings
        max_tokens = max_tokens or settings.llm_model_max_tokens
        temperature = temperature or settings.llm_model_temperature
        model_name = model or self.model_name
        
        # Format messages
        prompt = self._format_messages(messages)
        logger.debug(f"Generated prompt: {prompt[:100]}...")
        
        try:
            # Tokenize for token counting
            tokens = self.tokenizer.encode(prompt)
            prompt_tokens = len(tokens)
            
            # Generate with correct MLX-LM API
            start_time = time.time()
            logger.debug(f"Calling generate with max_tokens={max_tokens}, temperature={temperature}")
            
            # Try different parameter approaches based on MLX-LM version
            response_text = None
            error_details = []
            
            # Approach 1: Use 'temp' parameter (newer MLX-LM)
            try:
                response_text = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature  # MLX-LM uses 'temp' not 'temperature'
                )
                logger.debug("Successfully used generate with temp parameter")
            except TypeError as e:
                error_details.append(f"temp parameter: {e}")
                
                # Approach 2: Use 'temperature' parameter (older MLX-LM)
                try:
                    response_text = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    logger.debug("Successfully used generate with temperature parameter")
                except TypeError as e2:
                    error_details.append(f"temperature parameter: {e2}")
                    
                    # Approach 3: Positional arguments only
                    try:
                        response_text = generate(
                            self.model,
                            self.tokenizer,
                            prompt
                        )
                        logger.warning("Used basic generate call without sampling parameters")
                    except Exception as e3:
                        error_details.append(f"basic call: {e3}")
                        logger.error("All generate approaches failed: " + "; ".join(error_details))
                        raise RuntimeError(f"Failed to generate completion with all approaches: {'; '.join(error_details)}")
            
            if response_text is None:
                raise RuntimeError("Generate function returned None")
            
            generation_time = time.time() - start_time
            
            # Clean up response text (remove the prompt echo if present)
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            # Calculate completion token count
            completion_tokens_encoded = self.tokenizer.encode(response_text)
            completion_tokens = len(completion_tokens_encoded)
            total_tokens = prompt_tokens + completion_tokens
            
            logger.info(f"Generated completion in {generation_time:.2f}s, "
                       f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens}")
            
            # Create response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop"
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise RuntimeError(f"Failed to generate completion: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "model_name": self.model_name
        }


@lru_cache()
def get_mlx_service() -> MLXService:
    """Get cached MLX service instance."""
    return MLXService()


# Global service instance for backward compatibility
mlx_service = get_mlx_service()
