import streamlit as st
import time
import os
import random

# Try to import ML libraries with error handling
try:
    from llama_cpp import Llama
    import os

    ML_AVAILABLE = True
    print("Successfully imported llama-cpp-python")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install llama-cpp-python: pip install llama-cpp-python")
    ML_AVAILABLE = False
except Exception as e:
    print(f"Other error during import: {e}")
    ML_AVAILABLE = False

# Mock mode for deployment testing - enable if ML not available
MOCK_MODE = not ML_AVAILABLE

# Page config
st.set_page_config(page_title="History Chatbot", page_icon="🤖", layout="wide")

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
    if MOCK_MODE:
        st.warning(
            "🎭 Running in MOCK MODE - ML libraries not available or model loading failed"
        )
        return None, None, "mock"

    if not ML_AVAILABLE:
        st.error("❌ ML libraries (torch/transformers) are not available")
        st.info("Please install them with: pip install torch transformers")
        st.stop()

    try:
        # Use local GGUF model
        model_path = "phi-3-mini-4k-instruct.Q4_K_M.gguf"

        if not os.path.exists(model_path):
            st.error(f"❌ Local model file not found: {model_path}")
            st.info("Please ensure the GGUF model file is in the app directory")
            return None, None, "error"

        st.info(f"Loading local GGUF model: {model_path}")

        # Load GGUF model with llama-cpp-python - optimized for speed
        model = Llama(
            model_path=model_path,
            n_ctx=1024,  # Reduced context for faster inference
            n_threads=8,  # Use all CPU cores
            n_gpu_layers=-1,  # Use GPU if available
            n_batch=512,  # Larger batch size
            verbose=False,
            use_mmap=True,  # Memory mapping for faster loading
            use_mlock=False,  # Don't lock memory
        )

        st.success(f"✅ Local GGUF model loaded successfully")
        return model, None, "cpu"  # GGUF handles tokenization internally

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Falling back to mock mode...")
        return None, None, "mock"


def answer_history_question(model, tokenizer, device, text):
    if MOCK_MODE or model is None:
        # Mock history responses - just return a sample answer
        time.sleep(1)  # Simulate processing time

        # Simple mock responses
        mock_responses = [
            "This is a mock history answer for demo purposes.",
            "The historical event you asked about is complex and has multiple causes.",
            "History shows us that this topic involves many interconnected factors.",
            "Your history question has been processed in mock mode.",
        ]

        # Try to make it somewhat realistic
        words = text.split()
        if len(words) > 10:
            # For long questions, return a more detailed answer
            return "Based on historical records, " + random.choice(mock_responses)
        else:
            return random.choice(mock_responses)

    # Create a prompt for history questions using Phi-3
    prompt = f"<|user|>\nAs a history expert, please answer this question: {text}<|end|>\n<|assistant|>\n"

    # Use llama-cpp-python for inference
    try:
        response = model(
            prompt,
            max_tokens=150,  # Shorter responses for speed
            temperature=0.3,  # Less randomness for faster generation
            top_p=0.8,
            top_k=20,  # Limit vocabulary for speed
            stop=["<|end|>", "<|user|>"],
            echo=False,
            repeat_penalty=1.1,
        )

        # Extract the simplified text from response
        simplified = response["choices"][0]["text"].strip()

        # Clean up any remaining tokens
        if simplified.startswith("<|assistant|>"):
            simplified = simplified[len("<|assistant|>") :].strip()

        return simplified

    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, there was an error simplifying your text."


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your History Chatbot 🤖 Ask me anything about history and I'll help you understand it better!",
        }
    ]


# Load model
with st.spinner("🤖 Bot is starting up..."):
    model, tokenizer, device = load_model()

# Header
st.title("🤖 History Chatbot")
st.caption("Chat with me to explore and understand history!")

# Chat interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about history..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking about history..."):
            start_time = time.time()
            answer = answer_history_question(model, tokenizer, device, prompt)
            end_time = time.time()

        # Show answer
        st.write(f"**Answer:** {answer}")

        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original words", len(prompt.split()))
        with col2:
            st.metric("Response words", len(answer.split()))
        with col3:
            st.metric("Time", f"{end_time - start_time:.1f}s")

        # Add bot response to history
        bot_response = f"**Answer:** {answer}\n\n📊 Question: {len(prompt.split())} words → Answer: {len(answer.split())} words"
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Sidebar with examples and controls
with st.sidebar:
    st.header("🎯 Quick Examples")

    examples = [
        "What caused the fall of the Roman Empire?",
        "How did the Industrial Revolution change society?",
        "What were the main causes of World War I?",
    ]

    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}", key=f"ex{i}"):
            st.session_state.messages.append({"role": "user", "content": example})

            with st.spinner("🤖 Processing history question..."):
                answer = answer_history_question(model, tokenizer, device, example)

            bot_response = f"**Answer:** {answer}\n\n📊 Question: {len(example.split())} words → Answer: {len(answer.split())} words"
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm your History Chatbot 🤖 Ask me anything about history and I'll help you understand it better!",
            }
        ]
        st.rerun()

    # Show model info
    if MOCK_MODE:
        st.info("🎭 MOCK MODE\n💻 Simulated responses")
    else:
        st.info(f"🤖 Model: Phi-3-mini (GGUF)\n💻 Device: Local CPU")

    if os.path.exists("Spiece Model.model"):
        st.success("✅ SentencePiece model detected")
