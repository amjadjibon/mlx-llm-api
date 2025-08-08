import logging
from typing import Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from ...core.dependencies import get_mlx_service_dependency
from ...models import (
    AudioSpeechRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse,
    AudioVerboseResponse,
    ErrorDetail,
    ErrorResponse,
)
from ...services.mlx_service import MLXService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["OpenAI Audio"])


@router.post(
    "/audio/transcriptions",
    response_model=Union[AudioTranscriptionResponse, AudioVerboseResponse],
    summary="Create transcription",
    description="Transcribe audio to text (OpenAI-compatible)",
    responses={
        200: {"description": "Successful transcription"},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(..., description="Model to use for transcription"),
    language: Optional[str] = Form(None, description="Language of the audio"),
    prompt: Optional[str] = Form(
        None, description="Optional prompt to guide transcription"
    ),
    response_format: Optional[str] = Form(
        "json", description="Format of the transcript output"
    ),
    temperature: Optional[float] = Form(0.0, description="Sampling temperature"),
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> Union[AudioTranscriptionResponse, AudioVerboseResponse]:
    """Transcribe audio to text using mlx-whisper models (OpenAI audio API)."""
    try:
        logger.info(f"Processing transcription request for model: {model}")

        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="No audio file provided",
                    code="NO_FILE",
                ).model_dump(),
            )

        # Check file size (25MB limit like OpenAI)
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
        audio_data = await file.read()

        if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="File size too large. Maximum size is 25MB.",
                    code="FILE_TOO_LARGE",
                ).model_dump(),
            )

        # Validate audio format
        allowed_formats = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
        file_extension = f".{file.filename.split('.')[-1].lower()}"
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message=f"Unsupported audio format. Supported formats: {', '.join(allowed_formats)}",
                    code="UNSUPPORTED_FORMAT",
                ).model_dump(),
            )

        # Transcribe audio
        result = await mlx_service.transcribe_audio(
            audio_data=audio_data,
            model_name=model,
            language=language,
            prompt=prompt,
            temperature=temperature or 0.0,
            response_format=response_format or "json",
        )

        # Format response based on response_format
        if response_format == "verbose_json":
            return AudioVerboseResponse(
                task="transcribe",
                language=result.get("language", "en"),
                duration=result.get("duration", 0.0),
                text=result.get("text", ""),
                segments=result.get("segments", []),
                words=result.get("words", []),
            )
        elif response_format == "text":
            return Response(content=result.get("text", ""), media_type="text/plain")
        else:
            return AudioTranscriptionResponse(text=result.get("text", ""))

        logger.info("Audio transcription completed successfully")

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during transcription: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                type="service_error", message=str(e), code="TRANSCRIPTION_ERROR"
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                type="internal_error",
                message="An unexpected error occurred during transcription",
                code="TRANSCRIPTION_FAILED",
            ).model_dump(),
        )


@router.post(
    "/audio/translations",
    response_model=Union[AudioTranslationResponse, AudioVerboseResponse],
    summary="Create translation",
    description="Translate audio to English text (OpenAI-compatible)",
    responses={
        200: {"description": "Successful translation"},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(..., description="Model to use for translation"),
    prompt: Optional[str] = Form(
        None, description="Optional prompt to guide translation"
    ),
    response_format: Optional[str] = Form(
        "json", description="Format of the transcript output"
    ),
    temperature: Optional[float] = Form(0.0, description="Sampling temperature"),
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> Union[AudioTranslationResponse, AudioVerboseResponse]:
    """Translate audio to English text using mlx-whisper models (OpenAI audio API)."""
    try:
        logger.info(f"Processing translation request for model: {model}")

        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="No audio file provided",
                    code="NO_FILE",
                ).model_dump(),
            )

        # Check file size (25MB limit like OpenAI)
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
        audio_data = await file.read()

        if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="File size too large. Maximum size is 25MB.",
                    code="FILE_TOO_LARGE",
                ).model_dump(),
            )

        # Validate audio format
        allowed_formats = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
        file_extension = f".{file.filename.split('.')[-1].lower()}"
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message=f"Unsupported audio format. Supported formats: {', '.join(allowed_formats)}",
                    code="UNSUPPORTED_FORMAT",
                ).model_dump(),
            )

        # Translate audio
        result = await mlx_service.translate_audio(
            audio_data=audio_data,
            model_name=model,
            prompt=prompt,
            temperature=temperature or 0.0,
            response_format=response_format or "json",
        )

        # Format response based on response_format
        if response_format == "verbose_json":
            return AudioVerboseResponse(
                task="translate",
                language=result.get("language", "unknown"),
                duration=result.get("duration", 0.0),
                text=result.get("text", ""),
                segments=result.get("segments", []),
                words=result.get("words", []),
            )
        elif response_format == "text":
            return Response(content=result.get("text", ""), media_type="text/plain")
        else:
            return AudioTranslationResponse(text=result.get("text", ""))

        logger.info("Audio translation completed successfully")

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during translation: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                type="service_error", message=str(e), code="TRANSLATION_ERROR"
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                type="internal_error",
                message="An unexpected error occurred during translation",
                code="TRANSLATION_FAILED",
            ).model_dump(),
        )


@router.post(
    "/audio/speech",
    summary="Create speech",
    description="Generate audio from text using text-to-speech (OpenAI-compatible)",
    responses={
        200: {
            "description": "Generated audio file",
            "content": {"audio/wav": {}, "audio/mpeg": {}},
        },
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_speech(
    request: AudioSpeechRequest,
    mlx_service: MLXService = Depends(get_mlx_service_dependency),
) -> Response:
    """Generate audio from text using KittenTTS (OpenAI audio API)."""
    try:
        logger.info(f"Processing TTS request for model: {request.model}")

        # Validate input length (OpenAI limit is 4096 chars)
        if len(request.input) > 4096:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorDetail(
                    type="invalid_request",
                    message="Input text too long. Maximum length is 4096 characters.",
                    code="INPUT_TOO_LONG",
                ).model_dump(),
            )

        # Generate speech
        audio_data = await mlx_service.generate_speech(
            text=request.input,
            voice=request.voice or "expr-voice-2-f",
            response_format=request.response_format or "wav",
            speed=request.speed or 1.0,
        )

        # Set appropriate content type
        content_type = "audio/wav"
        if request.response_format == "mp3":
            content_type = "audio/mpeg"
        elif request.response_format == "opus":
            content_type = "audio/opus"
        elif request.response_format == "aac":
            content_type = "audio/aac"
        elif request.response_format == "flac":
            content_type = "audio/flac"

        logger.info("TTS generation completed successfully")
        return Response(content=audio_data, media_type=content_type)

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during TTS: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                type="service_error", message=str(e), code="TTS_ERROR"
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected error during TTS: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                type="internal_error",
                message="An unexpected error occurred during speech generation",
                code="TTS_FAILED",
            ).model_dump(),
        )
