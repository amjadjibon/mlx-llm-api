from fastapi import APIRouter
from .endpoints import chat, health

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/api")
api_router.include_router(chat.router, prefix="/api") 