from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging
import uuid
from ..config import Settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """Middleware for request/response logging with request ID tracking."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def __call__(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log request (info level in development, debug in production)
        if self.settings.is_development:
            logger.info(f"[{request_id}] {request.method} {request.url.path}")
        else:
            logger.debug(f"[{request_id}] {request.method} {request.url.path}")

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            if self.settings.is_development:
                logger.info(f"[{request_id}] {response.status_code} - {duration:.4f}s")
            else:
                logger.debug(f"[{request_id}] {response.status_code} - {duration:.4f}s")

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{request_id}] Error processing request: {e} - {duration:.4f}s"
            )
            raise


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    """Setup all middleware for the FastAPI app."""

    # Security middleware should be added first
    if not settings.is_development:
        # Only add trusted host middleware in production
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom logging middleware (added last so it's processed first)
    app.middleware("http")(RequestLoggingMiddleware(settings))
