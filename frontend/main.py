import json
import os
import tempfile
import time
from typing import List

import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# Configure Streamlit page
st.set_page_config(
    page_title="MLX ChatGPT Clone",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "available_models" not in st.session_state:
    st.session_state.available_models = []

if "client" not in st.session_state:
    st.session_state.client = None


def initialize_openai_client(
    base_url: str, api_key: str = "not-needed"
) -> openai.OpenAI:
    """Initialize OpenAI client with MLX API base URL."""
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def get_available_models(client: openai.OpenAI) -> List[str]:
    """Fetch available models from the API."""
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []


def transcribe_audio(client: openai.OpenAI, audio_file) -> str:
    """Transcribe audio using the API."""
    try:
        response = client.audio.transcriptions.create(
            model="mlx-community/whisper-large-v3-mlx", file=audio_file, response_format="json"
        )
        return response.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""


def generate_speech(
    client: openai.OpenAI, text: str, voice: str = "expr-voice-2-f"
) -> bytes:
    """Generate speech from text using the API."""
    try:
        response = client.audio.speech.create(
            model="kitten-tts-nano", voice=voice, input=text, response_format="wav"
        )
        return response.content
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return b""


# Sidebar configuration
with st.sidebar:
    st.title("🤖 MLX ChatGPT Clone")
    st.markdown("---")

    # API Configuration
    st.subheader("🔧 API Configuration")
    base_url = st.text_input(
        "API Base URL", value="http://localhost:8000/v1", help="MLX API base URL"
    )

    # Initialize client
    if st.button("Connect to API") or st.session_state.client is None:
        try:
            st.session_state.client = initialize_openai_client(base_url)
            st.session_state.available_models = get_available_models(
                st.session_state.client
            )
            st.success("✅ Connected to API")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

    # Model selection
    if st.session_state.available_models:
        st.subheader("🎯 Model Selection")
        st.session_state.selected_model = st.selectbox(
            "Choose Model",
            st.session_state.available_models,
            index=0
            if st.session_state.selected_model is None
            else st.session_state.available_models.index(
                st.session_state.selected_model
            )
            if st.session_state.selected_model in st.session_state.available_models
            else 0,
        )

        # Model parameters
        st.subheader("⚙️ Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 1, 2048, 256)
        top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.01)

    st.markdown("---")

    # Audio features
    st.subheader("🎵 Audio Features")

    # Voice selection for TTS
    voice_options = [
        "expr-voice-2-f",
        "expr-voice-2-m",
        "expr-voice-3-f",
        "expr-voice-3-m",
        "expr-voice-4-f",
        "expr-voice-4-m",
        "expr-voice-5-f",
        "expr-voice-5-m",
    ]
    selected_voice = st.selectbox("TTS Voice", voice_options)

    # Voice settings
    st.write("🎤 Voice input available in chat below")
    st.caption("Click the microphone button next to the message input")

    st.markdown("---")

    # Chat controls
    st.subheader("💬 Chat Controls")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💾 Export Chat"):
        if st.session_state.messages:
            chat_data = {
                "model": st.session_state.selected_model,
                "messages": st.session_state.messages,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.download_button(
                label="Download Chat JSON",
                data=json.dumps(chat_data, indent=2),
                file_name=f"chat_export_{int(time.time())}.json",
                mime="application/json",
            )

# Main chat interface
st.title("💬 Chat with MLX Models")

# Display connection status
if not st.session_state.client or not st.session_state.available_models:
    st.warning("⚠️ Please connect to the API and select a model from the sidebar.")
    st.stop()

# Display current model
st.info(f"🤖 Current Model: **{st.session_state.selected_model}**")

# Chat messages container
chat_container = st.container()

with chat_container:
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Add audio icon for transcribed messages
            if message.get("type") == "audio":
                st.caption("🎤 Transcribed from audio")

            # Add TTS for assistant messages
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 10])
                with col1:
                    if st.button("🔊", key=f"tts_{i}"):
                        if st.session_state.client:
                            with st.spinner("Generating speech..."):
                                audio_data = generate_speech(
                                    st.session_state.client,
                                    message["content"],
                                    selected_voice,
                                )
                                if audio_data:
                                    st.audio(audio_data, format="audio/wav")

# Initialize recording state
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None

# Custom chat input with voice button
def process_user_input(user_input: str, input_type: str = "text"):
    """Process user input and generate assistant response."""
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "type": input_type
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
        if input_type == "audio":
            st.caption("🎤 Transcribed from voice")

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Thinking..."):
                # Prepare messages for API
                api_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]

                # Make API call
                response = st.session_state.client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )

                full_response = response.choices[0].message.content

            # Display response with typewriter effect
            words = full_response.split()
            displayed_response = ""

            for word in words:
                displayed_response += word + " "
                message_placeholder.markdown(displayed_response + "▌")
                time.sleep(0.05)

            message_placeholder.markdown(full_response)

            # Add assistant message to session state
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error generating response: {e}")

# Custom chat input area with voice button - ChatGPT style
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

# Initialize input counter for unique keys
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# Create a ChatGPT-like input container
input_container = st.container()
with input_container:
    # Create columns for the ChatGPT-like layout: [text input | voice | send]
    col1, col2, col3 = st.columns([8, 1, 1], gap="small")
    
    with col1:
        # Text input with ChatGPT styling - use dynamic key to enable clearing
        user_input = st.text_input(
            label="Message",
            placeholder="Send a message...",
            key=f"chat_input_{st.session_state.input_counter}",
            label_visibility="collapsed",
            help="Type your message here or use voice input"
        )
    
    with col2:
        # Voice recording button - compact version
        audio_bytes = audio_recorder(
            text="",
            recording_color="#ff4b4b",
            neutral_color="#f0f2f6",
            icon_name="microphone",
            icon_size="1x",
            key="voice_input"
        )
        
        # Process recorded audio immediately
        if audio_bytes and st.session_state.client:
            with st.spinner("🎤 Transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file.flush()

                    with open(tmp_file.name, "rb") as audio_file:
                        transcribed_text = transcribe_audio(st.session_state.client, audio_file)

                    os.unlink(tmp_file.name)

                    if transcribed_text:
                        # Process voice input and increment counter to clear input
                        process_user_input(transcribed_text, "audio")
                        st.session_state.input_counter += 1
                        st.rerun()
    
    with col3:
        # Send button with improved styling
        send_button = st.button(
            "⬆", 
            key="send_button", 
            help="Send message (Enter)",
            disabled=not user_input.strip() if user_input else True
        )

st.markdown('</div>', unsafe_allow_html=True)

# Process text input when send button is clicked or Enter is pressed
if user_input and user_input.strip():  # Only process non-empty messages
    if send_button:  # Send button clicked
        process_user_input(user_input.strip(), "text")
        # Increment counter to create new input widget (effectively clearing)
        st.session_state.input_counter += 1
        st.rerun()
    elif user_input != st.session_state.get("last_input", ""):  # Enter key pressed (new input)
        st.session_state.last_input = user_input
        process_user_input(user_input.strip(), "text")
        # Increment counter to create new input widget (effectively clearing)
        st.session_state.input_counter += 1
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    Powered by MLX-LM API • Built with Streamlit • 
    <a href='https://github.com/anthropics/claude-code' target='_blank'>MLX ChatGPT Clone</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add enhanced CSS for ChatGPT-like interface
st.markdown(
    """
<style>
    .stApp {
        max-width: 1200px;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        background-color: #f7f7f8;
        border: 1px solid #d9d9e3;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
        outline: none;
    }
    
    /* Send button styling - ChatGPT style */
    .stButton > button[key="send_button"] {
        background-color: #000000;
        color: white;
        border: none;
        border-radius: 8px;
        width: 32px;
        height: 32px;
        margin-top: 8px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.2s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
    }
    
    .stButton > button[key="send_button"]:hover:not(:disabled) {
        background-color: #2d2d2d;
        transform: scale(1.05);
    }
    
    .stButton > button[key="send_button"]:disabled {
        background-color: #e5e5e5;
        color: #9ca3af;
        cursor: not-allowed;
        transform: none;
    }
    
    /* Voice button container */
    .audio-recorder-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 8px;
    }
    
    /* Chat messages styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .user-message {
        background-color: #f0f8ff;
        border-left: 4px solid #0066cc;
    }
    
    .assistant-message {
        background-color: #f8f8f8;
        border-left: 4px solid #10a37f;
    }
    
    /* General button styling */
    .stButton button {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* TTS button styling */
    .stButton > button[key*="tts_"] {
        background-color: #f0f2f6;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        padding: 0;
        font-size: 14px;
    }
    
    .stButton > button[key*="tts_"]:hover {
        background-color: #e5e7eb;
        color: #10a37f;
    }
    
    /* Sidebar styling */
    .sidebar .stSelectbox label {
        font-weight: bold;
        color: #374151;
    }
    
    /* Input container */
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 20px 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 20px;
    }
    
    /* Voice button specific styling */
    div[data-testid="stAudioRecorder"] > div {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 8px;
    }
    
    /* Improve text input width */
    .stTextInput {
        width: 100%;
    }
    
    /* Chat container styling */
    .main-chat-container {
        padding-bottom: 100px;
    }
</style>
""",
    unsafe_allow_html=True,
)
