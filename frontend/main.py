import json
import os
import tempfile
import time
from typing import List
import re
import html

import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# Configure Streamlit page
st.set_page_config(
    page_title="MLX Chat",
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


# Text preprocessing helpers for UI and TTS
def remove_think_blocks(text: str) -> str:
    """Remove <think>...</think> sections from text (case-insensitive)."""
    if not text:
        return text
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)


def remove_emojis(text: str) -> str:
    """Remove emojis and miscellaneous symbols to keep TTS clean."""
    if not text:
        return text
    emoji_pattern = re.compile(
        """
        [\U0001F600-\U0001F64F]  # emoticons
        |[\U0001F300-\U0001F5FF]  # symbols & pictographs
        |[\U0001F680-\U0001F6FF]  # transport & map symbols
        |[\U0001F700-\U0001F77F]  # alchemical symbols
        |[\U0001F780-\U0001F7FF]  # geometric shapes extended
        |[\U0001F800-\U0001F8FF]  # supplemental arrows-c
        |[\U0001F900-\U0001F9FF]  # supplemental symbols and pictographs
        |[\U0001FA00-\U0001FA6F]  # chess symbols etc.
        |[\U0001FA70-\U0001FAFF]  # symbols and pictographs extended-a
        |[\u2600-\u26FF]          # misc symbols
        |[\u2700-\u27BF]          # dingbats
        """,
        flags=re.VERBOSE,
    )
    return emoji_pattern.sub("", text)


def prepare_tts_text(text: str) -> str:
    """Prepare text for TTS: remove <think> blocks and emojis, normalize spaces."""
    text_without_think = remove_think_blocks(text)
    text_without_emojis = remove_emojis(text_without_think)
    return re.sub(r"\s+", " ", text_without_emojis).strip()


# Sidebar configuration
with st.sidebar:
    st.title("🤖 MLX Chat")
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
            if message["role"] == "assistant":
                # Render thinking box if present
                think_blocks = re.findall(r"<think>([\s\S]*?)</think>", message["content"], flags=re.IGNORECASE)
                if think_blocks:
                    think_text = "\n\n".join(think_blocks)
                    st.markdown(
                        f"""
                        <div class="thinking-box">
                            <div class="thinking-title">💭 Thinking</div>
                            <div class="thinking-content">{html.escape(think_text)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write(remove_think_blocks(message["content"]))
            else:
                st.write(message["content"])

            # Add audio icon for transcribed messages
            if message.get("type") == "audio":
                st.caption("🎤 Transcribed from audio")

            # Add TTS for assistant messages
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 10])
                with col1:
                    if st.button("🔊", key=f"tts_{i}", help="Play audio"):
                        if st.session_state.client:
                            tts_text = prepare_tts_text(message["content"]) or ""
                            if tts_text:
                                audio_data = generate_speech(
                                    st.session_state.client,
                                    tts_text,
                                    selected_voice,
                                )
                                if audio_data:
                                    # Store audio in session state with unique key
                                    audio_key = f"audio_{i}_{int(time.time())}"
                                    st.session_state[audio_key] = audio_data
                                    
                                    # Show minimal audio player that autoplays
                                    st.audio(audio_data, format="audio/wav", autoplay=True)
                                    
                                    # Clean up old audio data
                                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith('audio_') and k != audio_key]
                                    for k in keys_to_remove:
                                        del st.session_state[k]

# Initialize recording state and input counter
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

def process_user_input(user_input: str, input_type: str = "text"):
    """Process user input and generate assistant response."""
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "type": input_type
    })

    # Generate assistant response
    try:
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

        # Add assistant message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    except Exception as e:
        st.error(f"Error generating response: {e}")
        # Add error message to chat
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Sorry, I encountered an error: {e}"}
        )

# Fixed chat input area with proper proportions
st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)

# Create the input layout with proper proportions
input_col1, input_col2, input_col3 = st.columns([0.85, 0.08, 0.07], gap="small")

with input_col1:
    # Text input
    user_input = st.text_input(
        label="Message",
        placeholder="Type your message here...",
        key=f"chat_input_{st.session_state.input_counter}",
        label_visibility="collapsed",
        help="Type your message or use voice input"
    )

with input_col2:
    # Voice recording button
    audio_bytes = audio_recorder(
        text="",
        recording_color="#ef4444",
        neutral_color="#6b7280",
        icon_name="microphone",
        icon_size="1x",
        key=f"voice_input_{st.session_state.input_counter}"
    )

with input_col3:
    # Send button
    send_clicked = st.button(
        "⬆",
        key="send_button",
        help="Send message",
        disabled=not user_input or not user_input.strip(),
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Handle audio input
if audio_bytes and st.session_state.client:
    with st.spinner("🎤 Transcribing audio..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()

                with open(tmp_file.name, "rb") as audio_file:
                    transcribed_text = transcribe_audio(st.session_state.client, audio_file)

                os.unlink(tmp_file.name)

                if transcribed_text:
                    process_user_input(transcribed_text, "audio")
                    st.session_state.input_counter += 1
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing audio: {e}")

# Handle text input
if user_input and user_input.strip():
    if send_clicked:
        process_user_input(user_input.strip(), "text")
        st.session_state.input_counter += 1
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em; padding: 20px 0;'>
        Powered by MLX-LM API • Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

# Enhanced CSS for better proportions and modern UI
st.markdown(
    """
<style>
    /* Global app styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Chat input wrapper */
    .chat-input-wrapper {
        position: sticky;
        bottom: 0;
        background: linear-gradient(to top, white 70%, rgba(255,255,255,0.9) 90%, transparent);
        padding: 20px 0 10px 0;
        margin-top: 20px;
        z-index: 100;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 16px;
        height: 50px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
        outline: none;
        background-color: #ffffff;
    }
    
    /* Send button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10a37f 0%, #0f8b6b 100%);
        color: white;
        border: none;
        border-radius: 12px;
        height: 50px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(16, 163, 127, 0.2);
    }
    
    .stButton > button:hover:not(:disabled) {
        background: linear-gradient(135deg, #0f8b6b 0%, #0d7a5e 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(16, 163, 127, 0.3);
    }
    
    .stButton > button:disabled {
        background: #e9ecef;
        color: #6c757d;
        transform: none;
        box-shadow: none;
        cursor: not-allowed;
    }
    
    /* Audio recorder styling */
    [data-testid="stAudioRecorder"] {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50px;
    }
    
    [data-testid="stAudioRecorder"] > div {
        width: 50px !important;
        height: 50px !important;
        background: #f8f9fa !important;
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stAudioRecorder"] > div:hover {
        border-color: #10a37f !important;
        background: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(16, 163, 127, 0.2) !important;
    }
    
    /* TTS button styling */
    .stButton > button[key*="tts_"] {
        background: #f8f9fa;
        color: #495057;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        width: 40px;
        height: 40px;
        padding: 0;
        font-size: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stButton > button[key*="tts_"]:hover {
        background: #10a37f;
        color: white;
        border-color: #10a37f;
        transform: scale(1.05);
    }
    
    /* Chat messages styling */
    .stChatMessage {
        padding: 16px 20px;
        margin: 12px 0;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    [data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant message styling */
    [data-testid="assistant-message"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    /* Thinking box styling */
    .thinking-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .thinking-title {
        font-size: 14px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .thinking-content {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 13px;
        color: #6c757d;
        background: #ffffff;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        white-space: pre-wrap;
        line-height: 1.4;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    .stSidebar .stSelectbox label,
    .stSidebar .stTextInput label,
    .stSidebar .stSlider label {
        font-weight: 600;
        color: #495057;
    }
    
    /* Info and warning boxes */
    .stInfo {
        background: linear-gradient(135deg, #10a37f 0%, #0f8b6b 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: #212529;
        border-radius: 12px;
        border: none;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    /* Hide audio controls when autoplay is used */
    audio[autoplay] {
        height: 0;
        opacity: 0;
        pointer-events: none;
    }
    
    /* Column gap adjustments */
    .row-widget.stHorizontal > div {
        padding-left: 4px;
        padding-right: 4px;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stTextInput > div > div > input,
        .stButton > button,
        [data-testid="stAudioRecorder"] > div {
            height: 45px;
        }
        
        .stApp {
            padding: 0 1rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)
