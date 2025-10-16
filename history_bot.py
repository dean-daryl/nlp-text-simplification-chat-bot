import streamlit as st
import time
import os
import random

# Import required libraries
from llama_cpp import Llama

# Page config
st.set_page_config(page_title="Text Simplifier Bot", page_icon="🤖", layout="wide")

# Custom CSS for chat-like appearance
st.markdown(
    """
<style>
    .chat-container {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #f1f1f1;
        color: black;
        margin-right: auto;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        background-color: white;
        padding: 10px;
        border-top: 1px solid #ddd;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_path = "phi-3-mini-4k-instruct.Q4_K_M.gguf"

    if os.path.exists(model_path):
        try:
            st.info("Loading Phi-3 model...")
            model = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    else:
        st.error("📁 Phi-3 model file not found!")
        st.error(
            "Please ensure 'phi-3-mini-4k-instruct.Q4_K_M.gguf' is in the app directory"
        )
        st.stop()


def simplify_text(model, text):
    # Create prompt for Phi-3
    prompt = f"<|user|>\nSimplify this text to make it easier to understand: {text}<|end|>\n<|assistant|>\n"

    # Generate response
    response = model(
        prompt, max_tokens=512, stop=["<|end|>"], echo=False, temperature=0.3
    )

    return response["choices"][0]["text"].strip()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your Text Simplifier Bot 🤖 Send me any complex text and I'll make it easier to understand!",
        }
    ]


# Load model
with st.spinner("🤖 Bot is starting up..."):
    model = load_model()

# Header
st.title("🤖 Text Simplifier Bot")
st.caption("Chat with me to simplify your complex text!")

# Chat interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your complex text here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Simplifying your text..."):
            start_time = time.time()
            simplified = simplify_text(model, prompt)
            end_time = time.time()

        # Show simplified text
        st.write(f"**Simplified:** {simplified}")

        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original words", len(prompt.split()))
        with col2:
            st.metric("Simplified words", len(simplified.split()))
        with col3:
            st.metric("Time", f"{end_time - start_time:.1f}s")

        # Add bot response to history
        bot_response = f"**Simplified:** {simplified}\n\n📊 Original: {len(prompt.split())} words → Simplified: {len(simplified.split())} words"
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Sidebar with controls
with st.sidebar:
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm your Text Simplifier Bot 🤖 Send me any complex text and I'll make it easier to understand!",
            }
        ]
        st.rerun()

    # Show model info
    st.info("🤖 Model: Phi-3-mini-4k-instruct\n💻 CPU (GGUF)")

    if os.path.exists("Spiece Model.model"):
        st.success("✅ SentencePiece model detected")
