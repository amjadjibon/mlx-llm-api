from fastapi import Request, status, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from ..models import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Validation error: {exc.errors()}")

    error_detail = ErrorDetail(
        type="validation_error",
        message="Invalid request data",
        code="VALIDATION_ERROR",
        details={"validation_errors": exc.errors()},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(error=error_detail).model_dump(),
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(f"[{request_id}] HTTP {exc.status_code} error: {exc.detail}")

    # If detail is already a dict (from our error models), use it directly
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    # Otherwise, wrap in our error structure
    error_detail = ErrorDetail(
        type="http_error", message=str(exc.detail), code=f"HTTP_{exc.status_code}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=error_detail).model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {str(exc)}", exc_info=True)

    error_detail = ErrorDetail(
        type="internal_error",
        message="An internal server error occurred",
        code="INTERNAL_SERVER_ERROR",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error=error_detail).model_dump(),
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup custom exception handlers."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
