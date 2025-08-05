import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api.router import api_router
from .core.middleware import setup_middleware
from .core.exceptions import setup_exception_handlers
from .services.mlx_service import mlx_service


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="OpenAI compatible API for MLX-LM",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include API routes
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "MLX LLM API",
            "version": settings.app_version,
            "docs": "/docs"
        }
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        print(f"Starting {settings.app_name} v{settings.app_version}")
        
        # Load model if path is provided
        if settings.llm_model_directory:
            print(f"Loading model from: {settings.llm_model_directory}")
            llm_model_path = os.path.join(settings.llm_model_directory, settings.llm_model_name)
            success = await mlx_service.load_model(llm_model_path)
            if success:
                print("Model loaded successfully")
            else:
                print("Failed to load model")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        print("Shutting down MLX LLM API")
    
    return app


# Create the app instance
app = create_app()
