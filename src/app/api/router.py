from fastapi import APIRouter
from .endpoints import chat, health, completions, embeddings, models

# Create main API router
api_router = APIRouter()

# Include OpenAI-compatible endpoints (v1 prefix)
api_router.include_router(chat.router)          # /v1/chat/completions
api_router.include_router(completions.router)   # /v1/completions
api_router.include_router(embeddings.router)    # /v1/embeddings
api_router.include_router(models.router)        # /v1/models

# Include management endpoints (api prefix)
api_router.include_router(health.router, prefix="/api") 