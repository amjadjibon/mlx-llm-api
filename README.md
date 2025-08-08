# MLX-LM API

A production-ready OpenAI-compatible API server for Apple MLX models with full audio capabilities and Streamlit ChatGPT clone.

## Features

- 🚀 **FastAPI Backend** - OpenAI-compatible endpoints
- 🤖 **Streamlit ChatGPT Clone** - Full-featured chat interface
- 🎵 **Audio Integration** - Speech-to-text and text-to-speech
- 🏗️ **Clean Architecture** - Modular, scalable design
- 🔧 **Dynamic Models** - Auto-discovery and switching
- 📚 **Complete Documentation** - API docs and examples
- 🏥 **Production Ready** - Health checks, logging, middleware

## Project Structure

```
mlx-llm-api/
├── backend/               # FastAPI backend
│   ├── app/               # Core application
│   │   ├── api/endpoints/ # Individual API endpoints
│   │   ├── core/          # Core functionality
│   │   ├── services/      # Business logic
│   │   └── ...            # Config, models, etc.
│   └── main.py            # Backend entry point
├── streamlit_app.py       # ChatGPT clone
├── frontend/              # Additional frontend
└── ...                    # Config and docs
```

## Quick Start

### 1. Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Download Models
```bash
# Install huggingface-hub
pip install huggingface-hub

# Create models directory
mkdir -p models

# Download a recommended model
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit --local-dir models/Qwen2.5-7B-Instruct-4bit

# Download an embedding model
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit --local-dir models/all-MiniLM-L6-v2-4bit
```

### 3. Configure Environment
```bash
cp env.example .env
# Edit .env with your model directory path
echo "LLM_MODEL_DIRECTORY=./models" >> .env
```

### 4. Start Backend API
```bash
python backend/main.py
```

### 5. Launch Frontend App (Optional)
```bash
streamlit run frontend/main.py
# or: streamlit run streamlit_app.py
```

## Access Points

- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (development only)
- **Streamlit App**: http://localhost:8501
- **Health Check**: http://localhost:8000/api/health

## OpenAI-Compatible Endpoints

### Core API
- **Chat Completions**: `POST /v1/chat/completions`
- **Text Completions**: `POST /v1/completions`
- **Embeddings**: `POST /v1/embeddings`
- **Models**: `GET /v1/models`, `GET /v1/models/{id}`

### Audio API
- **Transcriptions**: `POST /v1/audio/transcriptions`
- **Translations**: `POST /v1/audio/translations`
- **Text-to-Speech**: `POST /v1/audio/speech`

### Management
- **Health Check**: `GET /api/health`
- **Model Management**: `POST /api/load-model`, `GET /api/models`

## Example Usage

```bash
# Chat with your MLX model
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-mlx-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Transcribe audio
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3"

# Generate speech
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "kitten-tts-nano", "input": "Hello world!", "voice": "expr-voice-2-f"}' \
  --output speech.wav
```

## Streamlit ChatGPT Clone

The included Streamlit app provides a complete ChatGPT alternative with:

- 💬 **Real-time chat** with your MLX models
- 🎤 **Voice recording** and transcription
- 🔊 **Text-to-speech** responses
- ⚙️ **Model selection** and parameter control
- 💾 **Chat export** and history management

Launch with: `streamlit run streamlit_app.py`

## Documentation

See [CLAUDE.md](CLAUDE.md) for complete documentation including:
- Detailed API examples in Python, JavaScript, and cURL
- Architecture overview and design decisions
- OpenAI library compatibility guide
- Audio features and configuration
- Production deployment guidelines

## License

MIT License
