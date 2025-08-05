import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .config import get_settings
from .api.router import api_router
from .core.middleware import setup_middleware
from .core.exceptions import setup_exception_handlers
from .services.mlx_service import get_mlx_service

# Configure logging
def setup_logging(log_level: str = "INFO"):
    """Setup application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan context manager."""
    settings = get_settings()
    mlx_service = get_mlx_service()
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize model directory and discover models
    if settings.llm_model_directory:
        logger.info(f"Setting model directory: {settings.llm_model_directory}")
        mlx_service.set_model_directory(settings.llm_model_directory)
        
        # Load default model if specified
        if settings.llm_model_name:
            logger.info(f"Loading default model: {settings.llm_model_name}")
            try:
                success = await mlx_service.ensure_model_loaded(settings.llm_model_name)
                if success:
                    logger.info(f"Default model loaded successfully: {settings.llm_model_name}")
                else:
                    logger.warning(f"Failed to load default model: {settings.llm_model_name}")
            except Exception as e:
                logger.error(f"Error loading default model: {e}")
        else:
            # Load first available model as default
            default_model = mlx_service.get_default_model()
            if default_model:
                logger.info(f"Loading first available model as default: {default_model}")
                try:
                    success = await mlx_service.ensure_model_loaded(default_model)
                    if success:
                        logger.info(f"Default model loaded successfully: {default_model}")
                    else:
                        logger.warning(f"Failed to load default model: {default_model}")
                except Exception as e:
                    logger.error(f"Error loading default model: {e}")
            else:
                logger.info("No models found in model directory")
        
        # Log available models
        available_models = mlx_service.get_available_models()
        logger.info(f"Available models: {list(available_models.keys())}")
    else:
        logger.info("No model directory configured - models can be loaded dynamically")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down MLX LLM API")
    if mlx_service.is_model_loaded():
        mlx_service.unload_model()
        logger.info("Model unloaded")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    # Setup logging first
    setup_logging(settings.log_level)
    
    # Create FastAPI app with lifespan management
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="OpenAI compatible API for MLX-LM models",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
        # OpenAPI metadata
        contact={
            "name": "MLX LLM API",
            "url": "https://github.com/your-repo/mlx-llm-api",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )
    
    # Setup middleware (order matters!)
    setup_middleware(app, settings)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include API routes
    app.include_router(api_router)
    
    # Root endpoint
    @app.get(
        "/",
        summary="API Root",
        description="Get basic information about the API",
        tags=["Root"]
    )
    async def _root():
        """Root endpoint returning basic API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "OpenAI compatible API for MLX-LM models",
            "docs_url": "/docs" if settings.is_development else None,
            "environment": settings.environment
        }
    
    # Health endpoint at root level for load balancers
    @app.get(
        "/ping",
        summary="Simple health check",
        description="Simple ping endpoint for load balancers",
        tags=["Root"]
    )
    async def _ping():
        """Simple ping endpoint."""
        return {"status": "ok"}
    
    return app


# Create the app instance
app = create_app()
