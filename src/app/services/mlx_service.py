from mlx_lm import load, generate
from typing import List, Optional, Dict, Any
import time
import uuid
import logging
import os
from pathlib import Path
from functools import lru_cache
from ..models import (
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from ..config import get_settings

logger = logging.getLogger(__name__)

# Import mlx-embeddings with error handling
try:
    from mlx_embeddings.utils import load as load_embedding_model

    MLX_EMBEDDINGS_AVAILABLE = True
    logger.info("mlx-embeddings library is available")
except ImportError:
    MLX_EMBEDDINGS_AVAILABLE = False
    logger.warning(
        "mlx-embeddings not available. Embeddings will use placeholder implementation."
    )


class MLXService:
    """Service for handling MLX model operations with dynamic model management."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.current_model_path: Optional[str] = None
        self.current_model_name: Optional[str] = None
        self.available_models: Dict[str, str] = {}  # model_name -> model_path
        self.model_directory: Optional[str] = None

        # Separate embedding model handling
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.embedding_model_loaded = False
        self.current_embedding_model_path: Optional[str] = None
        self.current_embedding_model_name: Optional[str] = None

    def discover_models(self, model_directory: str) -> Dict[str, str]:
        """Discover all available models in the specified directory and subdirectories."""
        models = {}

        if not os.path.exists(model_directory):
            logger.warning(f"Model directory does not exist: {model_directory}")
            return models

        try:
            model_dir_path = Path(model_directory)
            logger.info(f"Discovering models in: {model_directory}")

            # Look for models in the main directory and subdirectories
            self._scan_directory_for_models(model_dir_path, models)

            logger.info(f"Discovered {len(models)} models: {list(models.keys())}")

        except Exception as e:
            logger.error(f"Error discovering models in {model_directory}: {e}")

        return models

    def _scan_directory_for_models(
        self,
        directory: Path,
        models: Dict[str, str],
        max_depth: int = 2,
        current_depth: int = 0,
    ) -> None:
        """Recursively scan directory for models up to max_depth."""
        if current_depth > max_depth:
            return

        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Check if this directory itself is a model
                    if self._is_model_directory(item):
                        # Use a hierarchical name if nested
                        if current_depth > 0:
                            model_name = f"{directory.name}/{item.name}"
                        else:
                            model_name = item.name
                        models[model_name] = str(item)
                        logger.debug(f"Found model directory: {model_name} -> {item}")
                    else:
                        # Recurse into subdirectory to look for models
                        self._scan_directory_for_models(
                            item, models, max_depth, current_depth + 1
                        )

                elif item.is_file() and item.suffix in [
                    ".safetensors",
                    ".bin",
                    ".pth",
                    ".npz",
                ]:
                    # Single model file
                    if current_depth > 0:
                        model_name = f"{directory.name}/{item.stem}"
                    else:
                        model_name = item.stem
                    models[model_name] = str(item)
                    logger.debug(f"Found model file: {model_name} -> {item}")

        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {directory}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

    def _is_model_directory(self, path: Path) -> bool:
        """Check if a directory contains MLX model files."""
        try:
            files_in_dir = [f.name for f in path.iterdir() if f.is_file()]

            # Must have config.json for MLX models
            if "config.json" not in files_in_dir:
                return False

            # Check for model weight files
            has_model_weights = False

            # Check for exact matches
            for model_file in ["model.safetensors", "weights.npz", "pytorch_model.bin"]:
                if model_file in files_in_dir:
                    has_model_weights = True
                    break

            # Check for sharded models (model-00001-of-00002.safetensors format)
            if not has_model_weights:
                for file_in_dir in files_in_dir:
                    if (
                        file_in_dir.startswith("model-")
                        and ("-of-" in file_in_dir)
                        and (
                            file_in_dir.endswith(".safetensors")
                            or file_in_dir.endswith(".bin")
                        )
                    ):
                        has_model_weights = True
                        break

            # Should also have tokenizer files for text models
            has_tokenizer = any(
                f in files_in_dir
                for f in [
                    "tokenizer.json",
                    "tokenizer.model",
                    "vocab.txt",
                    "tokenizer_config.json",
                ]
            )

            # For text generation models, we expect both model weights and tokenizer
            # For other models (like embeddings), weights + config might be enough
            result = has_model_weights and (
                has_tokenizer or "sentence_bert_config.json" in files_in_dir
            )

            if result:
                logger.debug(
                    f"Valid model directory: {path.name} (weights: {has_model_weights}, tokenizer: {has_tokenizer})"
                )

            return result

        except Exception as e:
            logger.warning(f"Error checking model directory {path}: {e}")
            return False

    def set_model_directory(self, model_directory: str) -> None:
        """Set the model directory and discover available models."""
        self.model_directory = model_directory

        # Check if the path contains multiple model source directories
        model_dir_path = Path(model_directory)
        if model_dir_path.exists():
            # Look for common model source directories
            subdirs = [d for d in model_dir_path.iterdir() if d.is_dir()]
            model_source_dirs = [
                d
                for d in subdirs
                if d.name
                in ["lmstudio-community", "mlx-community", "huggingface", "models"]
            ]

            if model_source_dirs and not any(
                self._is_model_directory(d) for d in subdirs[:3]
            ):  # Check first few dirs
                logger.info(
                    f"Found model source directories: {[d.name for d in model_source_dirs]}"
                )
                # Scan each source directory
                all_models = {}
                for source_dir in model_source_dirs:
                    source_models = self.discover_models(str(source_dir))
                    # Prefix with source directory name to avoid conflicts
                    for model_name, model_path in source_models.items():
                        prefixed_name = (
                            f"{source_dir.name}/{model_name}"
                            if "/" not in model_name
                            else model_name
                        )
                        all_models[prefixed_name] = model_path
                self.available_models = all_models
            else:
                # Single directory scan
                self.available_models = self.discover_models(model_directory)
        else:
            logger.warning(f"Model directory does not exist: {model_directory}")
            self.available_models = {}

        logger.info(
            f"Set model directory to {model_directory} with {len(self.available_models)} models"
        )

    def get_available_models(self) -> Dict[str, str]:
        """Get all available models."""
        return self.available_models.copy()

    def get_default_model(self) -> Optional[str]:
        """Get the default model name (first available model)."""
        if not self.available_models:
            return None
        return next(iter(self.available_models.keys()))

    async def ensure_model_loaded(self, requested_model: str) -> bool:
        """Ensure the requested model is loaded, switching if necessary."""
        # If no model is requested, use current or default
        if not requested_model:
            if self.model_loaded:
                return True
            requested_model = self.get_default_model()
            if not requested_model:
                logger.error("No models available and no default model")
                return False

        # If the requested model is already loaded, do nothing
        if self.model_loaded and self.current_model_name == requested_model:
            logger.debug(f"Model {requested_model} is already loaded")
            return True

        # Check if the requested model is available
        if requested_model not in self.available_models:
            logger.error(
                f"Requested model '{requested_model}' not found in available models: {list(self.available_models.keys())}"
            )
            return False

        # Unload current model if loaded
        if self.model_loaded:
            logger.info(f"Unloading current model: {self.current_model_name}")
            self.unload_model()

        # Load the requested model
        model_path = self.available_models[requested_model]
        logger.info(f"Loading requested model: {requested_model} from {model_path}")
        return await self.load_model(model_path, requested_model)

    async def load_embedding_model(
        self, model_path: str, model_name: Optional[str] = None
    ) -> bool:
        """Load an embedding model specifically."""
        if not MLX_EMBEDDINGS_AVAILABLE:
            logger.warning("Cannot load embedding model: mlx-embeddings not available")
            return False

        try:
            logger.info(f"Loading MLX embedding model from: {model_path}")
            self.embedding_model, self.embedding_tokenizer = load_embedding_model(
                model_path
            )
            self.embedding_model_loaded = True
            self.current_embedding_model_path = model_path
            self.current_embedding_model_name = model_name or os.path.basename(
                model_path
            )
            logger.info(
                f"Successfully loaded embedding model: {self.current_embedding_model_name}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading embedding model from {model_path}: {e}")
            self.embedding_model_loaded = False
            self.current_embedding_model_path = None
            self.current_embedding_model_name = None
            return False

    def unload_embedding_model(self) -> None:
        """Unload the current embedding model."""
        if self.embedding_model_loaded:
            logger.info(
                f"Unloading embedding model: {self.current_embedding_model_name}"
            )
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.embedding_model_loaded = False
        self.current_embedding_model_path = None
        self.current_embedding_model_name = None

    async def ensure_embedding_model_loaded(self, requested_model: str) -> bool:
        """Ensure the requested embedding model is loaded."""
        # If the requested model is already loaded as embedding model, do nothing
        if (
            self.embedding_model_loaded
            and self.current_embedding_model_name == requested_model
        ):
            logger.debug(f"Embedding model {requested_model} is already loaded")
            return True

        # Check if the requested model is available
        if requested_model not in self.available_models:
            logger.error(
                f"Requested embedding model '{requested_model}' not found in available models"
            )
            return False

        # Check if this model is suitable for embeddings
        if not self.is_embedding_model(requested_model):
            logger.warning(
                f"Model '{requested_model}' may not be optimized for embeddings"
            )

        # Unload current embedding model if loaded
        if self.embedding_model_loaded:
            logger.info(
                f"Unloading current embedding model: {self.current_embedding_model_name}"
            )
            self.unload_embedding_model()

        # Load the requested embedding model
        model_path = self.available_models[requested_model]
        logger.info(
            f"Loading requested embedding model: {requested_model} from {model_path}"
        )
        return await self.load_embedding_model(model_path, requested_model)

    async def load_model(
        self, model_path: str, model_name: Optional[str] = None
    ) -> bool:
        """Load the MLX model."""
        try:
            logger.info(f"Loading MLX model from: {model_path}")
            self.model, self.tokenizer = load(model_path)
            self.model_loaded = True
            self.current_model_path = model_path
            self.current_model_name = model_name or os.path.basename(model_path)
            logger.info(f"Successfully loaded model: {self.current_model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            self.model_loaded = False
            self.current_model_path = None
            self.current_model_name = None
            return False

    def unload_model(self) -> None:
        """Unload the current model."""
        if self.model_loaded:
            logger.info(f"Unloading model: {self.current_model_name}")
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.current_model_path = None
        self.current_model_name = None

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for the model using tokenizer's chat template if available."""
        # Try to use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Convert messages to dict format for chat template
                message_dicts = [
                    {"role": msg.role, "content": msg.content} for msg in messages
                ]
                prompt = self.tokenizer.apply_chat_template(
                    message_dicts, add_generation_prompt=True, tokenize=False
                )
                logger.debug("Using tokenizer's chat template")
                return prompt
            except Exception as e:
                logger.warning(
                    f"Failed to use chat template, falling back to manual formatting: {e}"
                )

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
        model: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Generate a chat completion with dynamic model loading."""
        # Ensure the requested model is loaded
        requested_model = model or self.current_model_name or self.get_default_model()

        if not await self.ensure_model_loaded(requested_model):
            available_models = (
                list(self.available_models.keys())
                if self.available_models
                else ["No models available"]
            )
            raise RuntimeError(
                f"Failed to load model '{requested_model}'. Available models: {available_models}"
            )

        if not self.model_loaded:
            raise RuntimeError("No model loaded after ensure_model_loaded")

        settings = get_settings()

        # Use provided parameters or fallback to settings
        max_tokens = max_tokens or settings.llm_model_max_tokens
        temperature = temperature or settings.llm_model_temperature
        model_name = requested_model or self.current_model_name

        # Format messages
        prompt = self._format_messages(messages)
        logger.debug(f"Generated prompt: {prompt[:100]}...")

        try:
            # Tokenize for token counting
            tokens = self.tokenizer.encode(prompt)
            prompt_tokens = len(tokens)

            # Generate with correct MLX-LM API
            start_time = time.time()
            logger.debug(
                f"Calling generate with max_tokens={max_tokens}, temperature={temperature}"
            )

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
                    temp=temperature,  # MLX-LM uses 'temp' not 'temperature'
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
                        temperature=temperature,
                    )
                    logger.debug(
                        "Successfully used generate with temperature parameter"
                    )
                except TypeError as e2:
                    error_details.append(f"temperature parameter: {e2}")

                    # Approach 3: Positional arguments only
                    try:
                        response_text = generate(self.model, self.tokenizer, prompt)
                        logger.warning(
                            "Used basic generate call without sampling parameters"
                        )
                    except Exception as e3:
                        error_details.append(f"basic call: {e3}")
                        logger.error(
                            "All generate approaches failed: "
                            + "; ".join(error_details)
                        )
                        raise RuntimeError(
                            f"Failed to generate completion with all approaches: {'; '.join(error_details)}"
                        )

            if response_text is None:
                raise RuntimeError("Generate function returned None")

            generation_time = time.time() - start_time

            # Clean up response text (remove the prompt echo if present)
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt) :].strip()

            # Calculate completion token count
            completion_tokens_encoded = self.tokenizer.encode(response_text)
            completion_tokens = len(completion_tokens_encoded)
            total_tokens = prompt_tokens + completion_tokens

            logger.info(
                f"Generated completion in {generation_time:.2f}s, "
                f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"
            )

            # Create response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    async def generate_text_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> CompletionResponse:
        """Generate a text completion (OpenAI completions API)."""
        # Ensure the requested model is loaded
        requested_model = model or self.current_model_name or self.get_default_model()

        if not await self.ensure_model_loaded(requested_model):
            available_models = (
                list(self.available_models.keys())
                if self.available_models
                else ["No models available"]
            )
            raise RuntimeError(
                f"Failed to load model '{requested_model}'. Available models: {available_models}"
            )

        if not self.model_loaded:
            raise RuntimeError("No model loaded after ensure_model_loaded")

        settings = get_settings()

        # Use provided parameters or fallback to settings
        max_tokens = max_tokens or settings.llm_model_max_tokens
        temperature = temperature or settings.llm_model_temperature
        model_name = requested_model or self.current_model_name

        logger.debug(f"Generating text completion for prompt: {prompt[:100]}...")

        try:
            # Tokenize for token counting
            tokens = self.tokenizer.encode(prompt)
            prompt_tokens = len(tokens)

            # Generate with correct MLX-LM API
            start_time = time.time()
            logger.debug(
                f"Calling generate with max_tokens={max_tokens}, temperature={temperature}"
            )

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
                    temp=temperature,
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
                        temperature=temperature,
                    )
                    logger.debug(
                        "Successfully used generate with temperature parameter"
                    )
                except TypeError as e2:
                    error_details.append(f"temperature parameter: {e2}")

                    # Approach 3: Positional arguments only
                    try:
                        response_text = generate(self.model, self.tokenizer, prompt)
                        logger.warning(
                            "Used basic generate call without sampling parameters"
                        )
                    except Exception as e3:
                        error_details.append(f"basic call: {e3}")
                        logger.error(
                            "All generate approaches failed: "
                            + "; ".join(error_details)
                        )
                        raise RuntimeError(
                            f"Failed to generate completion with all approaches: {'; '.join(error_details)}"
                        )

            if response_text is None:
                raise RuntimeError("Generate function returned None")

            generation_time = time.time() - start_time

            # Clean up response text (remove the prompt echo if present)
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt) :].strip()

            # Calculate completion token count
            completion_tokens_encoded = self.tokenizer.encode(response_text)
            completion_tokens = len(completion_tokens_encoded)
            total_tokens = prompt_tokens + completion_tokens

            logger.info(
                f"Generated text completion in {generation_time:.2f}s, "
                f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"
            )

            # Create response
            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    CompletionChoice(text=response_text, index=0, finish_reason="stop")
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )

        except Exception as e:
            logger.error(f"Error generating text completion: {e}")
            raise RuntimeError(f"Failed to generate text completion: {str(e)}")

    async def generate_embeddings(
        self, input_texts: List[str], model: Optional[str] = None
    ) -> EmbeddingResponse:
        """Generate embeddings for input texts using mlx-embeddings."""
        # Determine the requested model
        requested_model = (
            model or self._get_best_embedding_model() or self.get_default_model()
        )

        if not requested_model:
            raise RuntimeError("No embedding model available")

        # Use mlx-embeddings if available and model is suitable for embeddings
        if MLX_EMBEDDINGS_AVAILABLE and self.is_embedding_model(requested_model):
            return await self._generate_embeddings_with_mlx(
                input_texts, requested_model
            )
        else:
            # Fall back to text generation model approach (placeholder)
            return await self._generate_embeddings_fallback(
                input_texts, requested_model
            )

    async def _generate_embeddings_with_mlx(
        self, input_texts: List[str], requested_model: str
    ) -> EmbeddingResponse:
        """Generate embeddings using mlx-embeddings library."""
        # Ensure the embedding model is loaded
        if not await self.ensure_embedding_model_loaded(requested_model):
            available_models = (
                list(self.available_models.keys())
                if self.available_models
                else ["No models available"]
            )
            raise RuntimeError(
                f"Failed to load embedding model '{requested_model}'. Available models: {available_models}"
            )

        if not self.embedding_model_loaded:
            raise RuntimeError(
                "No embedding model loaded after ensure_embedding_model_loaded"
            )

        logger.info(
            f"Generating embeddings for {len(input_texts)} texts using mlx-embeddings"
        )

        try:
            embeddings_data = []
            total_tokens = 0

            for i, text in enumerate(input_texts):
                # Encode text and get embeddings using mlx-embeddings
                input_ids = self.embedding_tokenizer.encode(text, return_tensors="mlx")

                # Count tokens for usage tracking
                total_tokens += len(input_ids)

                # Generate embeddings
                outputs = self.embedding_model(input_ids)

                # Extract embedding vector (mean pooled and normalized)
                if hasattr(outputs, "text_embeds"):
                    embedding = outputs.text_embeds.tolist()
                elif hasattr(outputs, "last_hidden_state"):
                    # Mean pooling for models without text_embeds attribute
                    embedding = outputs.last_hidden_state.mean(axis=1).tolist()
                else:
                    raise RuntimeError("Could not extract embeddings from model output")

                # Ensure embedding is a flat list
                if isinstance(embedding[0], list):
                    embedding = embedding[0]

                embeddings_data.append(
                    EmbeddingData(object="embedding", embedding=embedding, index=i)
                )

            logger.info(f"Successfully generated {len(embeddings_data)} embeddings")

            return EmbeddingResponse(
                object="list",
                data=embeddings_data,
                model=requested_model,
                usage=EmbeddingUsage(
                    prompt_tokens=total_tokens, total_tokens=total_tokens
                ),
            )

        except Exception as e:
            logger.error(f"Error generating embeddings with mlx-embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    async def _generate_embeddings_fallback(
        self, input_texts: List[str], requested_model: str
    ) -> EmbeddingResponse:
        """Fallback embedding generation using text generation models."""
        # Ensure the text generation model is loaded
        if not await self.ensure_model_loaded(requested_model):
            available_models = (
                list(self.available_models.keys())
                if self.available_models
                else ["No models available"]
            )
            raise RuntimeError(
                f"Failed to load model '{requested_model}'. Available models: {available_models}"
            )

        logger.warning(
            f"Using fallback embedding generation for model: {requested_model}"
        )

        # Placeholder implementation for text generation models
        embeddings_data = []
        total_tokens = 0

        for i, text in enumerate(input_texts):
            # Count tokens for usage tracking
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)

            # Generate a deterministic embedding based on text content
            import hashlib

            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Create a deterministic embedding vector
            embedding = []
            for j in range(384):  # Standard embedding size
                # Use hash to create deterministic values
                hash_val = int(
                    text_hash[(j * 2) % len(text_hash) : (j * 2 + 2) % len(text_hash)],
                    16,
                )
                normalized_val = (hash_val / 255.0) * 2 - 1  # Normalize to [-1, 1]
                embedding.append(normalized_val)

            embeddings_data.append(
                EmbeddingData(object="embedding", embedding=embedding, index=i)
            )

        return EmbeddingResponse(
            object="list",
            data=embeddings_data,
            model=requested_model,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    def _get_best_embedding_model(self) -> Optional[str]:
        """Get the best available embedding model."""
        for model_name in self.available_models.keys():
            if self.is_embedding_model(model_name):
                return model_name
        return None

    def is_embedding_model(self, model_name: str) -> bool:
        """Check if a model is designed for embeddings."""
        embedding_indicators = [
            "embed",
            "embedding",
            "sentence",
            "minilm",
            "bert",
            "roberta",
        ]
        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in embedding_indicators)

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and available models."""
        return {
            "model_loaded": self.model_loaded,
            "current_model_path": self.current_model_path,
            "current_model_name": self.current_model_name,
            "model_directory": self.model_directory,
            "available_models": list(self.available_models.keys()),
            "total_available_models": len(self.available_models),
        }

    # ============================================================================
    # Audio Processing Methods
    # ============================================================================

    async def transcribe_audio(
        self,
        audio_data: bytes,
        model_name: str = "whisper-large-v3",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Transcribe audio using mlx-whisper."""
        try:
            import mlx_whisper
            import tempfile
            import os

            logger.info(f"Starting audio transcription with model: {model_name}")

            # Write audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            try:
                # Transcribe using mlx-whisper
                result = mlx_whisper.transcribe(
                    temp_audio_path,
                    path_or_hf_repo=model_name,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    verbose=response_format == "verbose_json",
                )

                logger.info("Audio transcription completed successfully")
                return result

            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

        except ImportError:
            logger.error("mlx-whisper not available for audio transcription")
            raise RuntimeError(
                "mlx-whisper library is required for audio transcription"
            )
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise RuntimeError(f"Audio transcription failed: {e}")

    async def translate_audio(
        self,
        audio_data: bytes,
        model_name: str = "whisper-large-v3",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """Translate audio to English using mlx-whisper."""
        try:
            import mlx_whisper
            import tempfile
            import os

            logger.info(f"Starting audio translation with model: {model_name}")

            # Write audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            try:
                # Translate using mlx-whisper (task='translate')
                result = mlx_whisper.transcribe(
                    temp_audio_path,
                    path_or_hf_repo=model_name,
                    task="translate",  # Force translation to English
                    initial_prompt=prompt,
                    temperature=temperature,
                    verbose=response_format == "verbose_json",
                )

                logger.info("Audio translation completed successfully")
                return result

            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

        except ImportError:
            logger.error("mlx-whisper not available for audio translation")
            raise RuntimeError("mlx-whisper library is required for audio translation")
        except Exception as e:
            logger.error(f"Audio translation failed: {e}")
            raise RuntimeError(f"Audio translation failed: {e}")

    async def generate_speech(
        self,
        text: str,
        voice: str = "expr-voice-2-f",
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """Generate speech from text using KittenTTS."""
        try:
            from kittentts import KittenTTS
            import soundfile as sf
            import io

            logger.info(f"Starting TTS generation with voice: {voice}")

            # Initialize KittenTTS model
            if not hasattr(self, "_tts_model") or self._tts_model is None:
                logger.info("Loading KittenTTS model...")
                self._tts_model = KittenTTS("KittenML/kitten-tts-nano-0.1")
                logger.info("KittenTTS model loaded successfully")

            # Generate audio
            audio = self._tts_model.generate(text, voice=voice)

            # Apply speed adjustment if needed
            if speed != 1.0:
                # Simple time-stretching by resampling (requires librosa)
                try:
                    import librosa

                    audio = librosa.effects.time_stretch(audio, rate=speed)
                except ImportError:
                    logger.warning("librosa not available, speed adjustment skipped")

            # Convert to bytes based on response format
            if response_format.lower() in ["wav", "pcm"]:
                # Write to bytes buffer as WAV
                buffer = io.BytesIO()
                sf.write(buffer, audio, 24000, format="wav")
                return buffer.getvalue()
            elif response_format.lower() == "mp3":
                # Convert to MP3 (requires pydub and ffmpeg)
                try:
                    from pydub import AudioSegment
                    import tempfile

                    # Write to temporary WAV file first
                    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
                        sf.write(temp_wav.name, audio, 24000, format="wav")

                        # Convert to MP3
                        audio_segment = AudioSegment.from_wav(temp_wav.name)
                        buffer = io.BytesIO()
                        audio_segment.export(buffer, format="mp3")
                        return buffer.getvalue()

                except ImportError:
                    logger.warning("pydub not available, falling back to WAV format")
                    buffer = io.BytesIO()
                    sf.write(buffer, audio, 24000, format="wav")
                    return buffer.getvalue()
            else:
                # Default to WAV
                buffer = io.BytesIO()
                sf.write(buffer, audio, 24000, format="wav")
                return buffer.getvalue()

        except ImportError as e:
            logger.error(f"Required library not available for TTS: {e}")
            raise RuntimeError(f"TTS library not available: {e}")
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise RuntimeError(f"TTS generation failed: {e}")

    def is_audio_model(self, model_name: str) -> bool:
        """Check if a model is designed for audio processing."""
        audio_indicators = ["whisper", "audio", "speech", "asr", "stt", "tts"]
        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in audio_indicators)


@lru_cache()
def get_mlx_service() -> MLXService:
    """Get cached MLX service instance."""
    return MLXService()


# Global service instance for backward compatibility
mlx_service = get_mlx_service()
