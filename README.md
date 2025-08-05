# MLX LLM API

A FastAPI-based OpenAI-compatible API for MLX-LM models.

## Features

- 🚀 FastAPI with Uvicorn server
- 🏗️ Clean, modular architecture
- 🔧 Configurable via environment variables
- 📚 Auto-generated API documentation
- 🏥 Health check endpoints
- 🔒 CORS and security middleware
- 📝 Comprehensive logging
- 🎯 OpenAI-compatible chat completions

## Architecture

```
src/
├── app/
│   ├── api/                 # API layer
│   │   ├── endpoints/       # Route handlers
│   │   └── router.py        # Main API router
│   ├── core/               # Core functionality
│   │   ├── middleware.py   # Custom middleware
│   │   └── exceptions.py   # Exception handlers
│   ├── services/           # Business logic
│   │   └── mlx_service.py  # MLX model operations
│   ├── config.py           # Configuration management
│   ├── models.py           # Pydantic models
│   └── application.py      # FastAPI app factory
└── main.py                 # Application entry point
```

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and configure it:

```bash
cp env.example .env
```

Edit `.env` with your settings:

```env
MODEL_PATH=/path/to/your/mlx/model
HOST=0.0.0.0
PORT=8000
```

### 3. Run the Application

```bash
# Development mode
python src/main.py

# Or using uvicorn directly
uvicorn src.app.application:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **Root Endpoint**: http://localhost:8000/

## API Endpoints

### Health & Management

- `GET /api/health` - Health check
- `POST /api/load-model` - Load a model
- `GET /api/models` - List available models

### Chat Completions

- `POST /api/v1/chat/completions` - Create chat completion
- `POST /api/v1/chat/completions/stream` - Streaming chat completion (TODO)

### Example Chat Completion Request

```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Configuration

The application can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | "MLX LLM API" | Application name |
| `APP_VERSION` | "0.1.0" | Application version |
| `DEBUG` | false | Debug mode |
| `HOST` | "0.0.0.0" | Server host |
| `PORT` | 8000 | Server port |
| `RELOAD` | false | Auto-reload on changes |
| `MODEL_PATH` | None | Path to MLX model |
| `MAX_TOKENS` | 1000 | Default max tokens |
| `TEMPERATURE` | 0.7 | Default temperature |

## Development

### Project Structure

- **Layered Architecture**: Separation of concerns with API, services, and core layers
- **Dependency Injection**: Services are injected where needed
- **Error Handling**: Centralized exception handling
- **Logging**: Structured logging with middleware
- **Configuration**: Environment-based configuration with Pydantic

### Adding New Endpoints

1. Create a new file in `src/app/api/endpoints/`
2. Define your router with endpoints
3. Include the router in `src/app/api/router.py`

### Adding New Services

1. Create a new file in `src/app/services/`
2. Implement your service logic
3. Import and use in your endpoints

## Production Deployment

For production deployment:

1. Set `DEBUG=false`
2. Configure proper CORS origins
3. Set trusted hosts
4. Use a production WSGI server
5. Configure proper logging
6. Set up monitoring and health checks

## License

MIT License
