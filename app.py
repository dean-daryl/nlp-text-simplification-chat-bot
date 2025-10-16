import streamlit as st
import time
import os
import random

# Try to import ML libraries with error handling
try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Mock mode for deployment testing
MOCK_MODE = not ML_AVAILABLE

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
    if MOCK_MODE:
        st.warning("🎭 Running in MOCK MODE - using simulated responses for demo")
        return None, None, "mock"

    try:
        # Use distilled model for faster inference
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def simplify_text(model, tokenizer, device, text):
    if MOCK_MODE:
        # Mock simplification - just return a simplified version
        time.sleep(1)  # Simulate processing time

        # Simple mock responses
        mock_responses = [
            "This text has been simplified using mock mode.",
            "The complex text was made easier to understand.",
            "This is a simpler version of your text.",
            "Your text was simplified successfully.",
        ]

        # Try to make it somewhat realistic
        words = text.split()
        if len(words) > 20:
            # For long text, return a shorter version
            simplified_words = words[: len(words) // 2] + ["and", "more."]
            return " ".join(simplified_words)
        else:
            return random.choice(mock_responses)

    input_text = f"simplify: {text}"
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", max_length=512, truncation=True
    )
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_length=256, num_beams=4, early_stopping=True, do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
    model, tokenizer, device = load_model()

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
            simplified = simplify_text(model, tokenizer, device, prompt)
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

# Sidebar with examples and controls
with st.sidebar:
    st.header("🎯 Quick Examples")

    examples = [
        "The implementation of sophisticated algorithms requires extensive computational resources.",
        "Contemporary research demonstrates remarkable progress in natural language processing.",
        "The proliferation of renewable energy technologies mitigates environmental degradation.",
    ]

    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}", key=f"ex{i}"):
            st.session_state.messages.append({"role": "user", "content": example})

            with st.spinner("🤖 Processing example..."):
                simplified = simplify_text(model, tokenizer, device, example)

            bot_response = f"**Simplified:** {simplified}\n\n📊 Original: {len(example.split())} words → Simplified: {len(simplified.split())} words"
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
                "content": "Hi! I'm your Text Simplifier Bot 🤖 Send me any complex text and I'll make it easier to understand!",
            }
        ]
        st.rerun()

    # Show model info
    if MOCK_MODE:
        st.info("🎭 MOCK MODE\n💻 Simulated responses")
    else:
        st.info(f"🤖 Model: FLAN-T5-small\n💻 Device: {device}")

    if os.path.exists("Spiece Model.model"):
        st.success("✅ SentencePiece model detected")
