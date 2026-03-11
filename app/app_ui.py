import streamlit as st
import os
import sys

# Add the project root to sys.path to allow importing from 'agents' and 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.search_agent import search_stream
from agents.rag_agent import build_vector_store, ask_stream, get_retrieved_chunks
from agents.ocr_agent import perform_ocr
from agents.image_agent import generate_image
from utils.db import init_db, save_message, load_history, clear_agent_history

# ---- Page Config ----
st.set_page_config(page_title="Agent Wolf", page_icon="🐺", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background and gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }

    /* Custom Header */
    .wolf-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .wolf-subtitle {
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Message Bubbles */
    .stChatMessage {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        padding: 1.5rem !important;
    }

    /* Agent Badges */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .badge-search { background-color: rgba(99, 102, 241, 0.2); color: #818cf8; border-color: #6366f1; }
    .badge-rag { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border-color: #22c55e; }
    .badge-vision { background-color: rgba(234, 179, 8, 0.2); color: #facc15; border-color: #eab308; }
    .badge-image { background-color: rgba(236, 72, 153, 0.2); color: #f472b6; border-color: #ec4899; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .sidebar-info {
        font-size: 0.85em;
        color: #94a3b8;
        padding: 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        text-align: center;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
    }

    /* Expander */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wolf-title">🐺 Agent Wolf</div>', unsafe_allow_html=True)
st.markdown('<div class="wolf-subtitle">Elite Research & Vision Assistant ✨</div>', unsafe_allow_html=True)


def render_chunks(chunks):
    """Renders retrieved chunks with relevance scores and metadata."""
    for i, chunk in enumerate(chunks, 1):
        relevance = chunk.get("relevance", 0)
        score = chunk.get("score", 0)
        source = chunk.get("source", "Unknown")
        page = chunk.get("page")

        # Color based on relevance
        if relevance >= 60:
            color_class = "relevance-high"
        elif relevance >= 40:
            color_class = "relevance-mid"
        else:
            color_class = "relevance-low"

        # Header with chunk number, relevance, and metadata
        page_info = f" | Page {page + 1}" if page is not None else ""
        st.markdown(
            f'<div class="chunk-header">'
            f'<strong>Chunk {i}</strong>'
            f'<span class="chunk-meta">'
            f'<span class="{color_class}">Relevance: {relevance}%</span>'
            f' | Distance: {score:.3f}'
            f'{page_info}'
            f' | Source: {source}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        st.code(chunk["content"], language=None)
        if i < len(chunks):
            st.divider()


# ---- Sidebar ----
with st.sidebar:
    st.header("🎚️ Control Center")

    st.markdown('<p class="sidebar-info">🐺 Agent Wolf Core <br> Powered by Groq Llama 3.3</p>', unsafe_allow_html=True)

    agent_choice = st.radio("Select Capability:", ["🌐 Global Search", "📚 Knowledge Base (RAG)", "👁️ Vision Analysis", "🎨 Image Generation"], index=0)

    # Map the display names back to logic names
    agent_logic_map = {
        "🌐 Global Search": "Search",
        "📚 Knowledge Base (RAG)": "RAG",
        "👁️ Vision Analysis": "Vision (OCR)",
        "🎨 Image Generation": "Image"
    }
    agent_choice_logic = agent_logic_map[agent_choice]

    st.divider()

    # Show file uploader only when RAG is selected
    if agent_choice_logic == "RAG":
        st.subheader("📥 Upload Sources")
        uploaded_files = st.file_uploader(
            "Feed documents to Wolf", type=["pdf", "txt"], accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🚀 Process Intelligence"):
                with st.spinner("Analyzing neural pathways..."):
                    try:
                        st.session_state.vector_store = build_vector_store(uploaded_files)
                        st.success(f"Wolf assimilated {len(uploaded_files)} sources! ✅")
                    except Exception as e:
                        st.error(f"Brain malfunction: {e}")

    # Show vision upload only when Vision is selected
    if agent_choice_logic == "Vision (OCR)":
        st.subheader("📷 Image Intel")
        uploaded_image = st.file_uploader(
            "Show Wolf an image", type=["png", "jpg", "jpeg"]
        )
        if uploaded_image:
            if st.button("🔍 Extract Vision"):
                with st.spinner("Enhancing image resolution..."):
                    try:
                        extracted_text = perform_ocr(uploaded_image.read())
                        st.session_state.vision_messages.append({"role": "user", "content": f"📷 Analyzed: {uploaded_image.name}"})
                        st.session_state.vision_messages.append({"role": "assistant", "content": extracted_text, "agent": "Vision"})
                        
                        save_message("Vision", "user", f"📷 Analyzed: {uploaded_image.name}")
                        save_message("Vision", "assistant", extracted_text, {"agent": "Vision"})
                        
                        st.success("Vision extraction complete! ✨")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Vision failure: {e}")

    # Show which agent is active
    if agent_choice_logic == "Search":
        st.info("🌐 Wolf is scanning the World Wide Web.")
    elif agent_choice_logic == "RAG":
        if "vector_store" in st.session_state:
            st.success("📚 Knowledge assimilated and ready.")
        else:
            st.warning("⚠️ Feed Wolf some documents first.")
    elif agent_choice_logic == "Vision (OCR)":
        st.info("👁️ Vision sensors active.")
    else:
        st.info("🎨 Image generation core online.")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Wipe Session"):
        st.session_state.search_messages = []
        st.session_state.rag_messages = []
        st.session_state.vision_messages = []
        st.session_state.image_messages = []
        clear_agent_history("Search")
        clear_agent_history("RAG")
        clear_agent_history("Vision")
        clear_agent_history("Image")
        st.rerun()

    # Export chat button
    current_key = f"{agent_choice_logic.lower()}_messages"
    if current_key == "vision (ocr)_messages": current_key = "vision_messages"
    
    if current_key in st.session_state and st.session_state[current_key]:
        chat_export = ""
        for msg in st.session_state[current_key]:
            role = "You" if msg["role"] == "user" else f"Agent Wolf ({msg.get('agent', agent_choice_logic)})"
            chat_export += f"**{role}:**\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "💾 Export Intelligence",
            data=chat_export,
            file_name=f"wolf_intel_{agent_choice_logic.lower()}.md",
            mime="text/markdown",
        )

# ---- Initialize Chat History (per agent) ----
if "search_messages" not in st.session_state:
    st.session_state.search_messages = []
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "vision_messages" not in st.session_state:
    st.session_state.vision_messages = []
if "image_messages" not in st.session_state:
    st.session_state.image_messages = []

# Initialize Database
init_db()

# ---- Initialize Chat History (per agent) ----
if "search_messages" not in st.session_state or not st.session_state.search_messages:
    st.session_state.search_messages = load_history("Search")
if "rag_messages" not in st.session_state or not st.session_state.rag_messages:
    st.session_state.rag_messages = load_history("RAG")
if "vision_messages" not in st.session_state or not st.session_state.vision_messages:
    st.session_state.vision_messages = load_history("Vision")
if "image_messages" not in st.session_state or not st.session_state.image_messages:
    st.session_state.image_messages = load_history("Image")

# ---- Persistent Vector Store Loading ----
if "vector_store" not in st.session_state and os.path.exists("database/faiss_index"):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        st.session_state.vector_store = FAISS.load_local(
            "database/faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading persistent intelligence: {e}")

# Pick the right history based on agent
if agent_choice_logic == "Search":
    messages = st.session_state.search_messages
elif agent_choice_logic == "RAG":
    messages = st.session_state.rag_messages
elif agent_choice_logic == "Image":
    messages = st.session_state.image_messages
else:
    messages = st.session_state.vision_messages

# ---- Display Chat History ----
for message in messages:
    with st.chat_message(message["role"]):
        # Show agent badge on assistant messages
        if message["role"] == "assistant":
            badge_class = "badge-search" if message.get("agent") == "Search" else ("badge-rag" if message.get("agent") == "RAG" else ("badge-image" if message.get("agent") == "Image" else "badge-vision"))
            badge_label = f"🐺 {message.get('agent', agent_choice_logic)}"
            st.markdown(
                f'<span class="agent-badge {badge_class}">{badge_label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])

        if message.get("image"):
            st.image(message["image"])

        # Show retrieved chunks if available (RAG)
        if "metadata" in message and message["metadata"].get("chunks"):
            with st.expander("📂 Source Intelligence Fragments"):
                render_chunks(message["metadata"]["chunks"])

# ---- Handle User Input ----
if prompt := st.chat_input("Command Agent Wolf..."):
    # Show user message
    messages.append({"role": "user", "content": prompt})
    save_message(agent_choice_logic, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate streamed response
    with st.chat_message("assistant"):
        # Show agent badge
        badge_class = "badge-search" if agent_choice_logic == "Search" else ("badge-rag" if agent_choice_logic == "RAG" else ("badge-image" if agent_choice_logic == "Image" else "badge-vision"))
        st.markdown(
            f'<span class="agent-badge {badge_class}">🐺 {agent_choice_logic}</span>',
            unsafe_allow_html=True,
        )

        try:
            if agent_choice_logic == "Image":
                gen_image = generate_image(prompt)
                st.image(gen_image)
                success_msg = f"Painted: {prompt}"
                messages.append({"role": "assistant", "content": success_msg, "agent": "Image", "image": gen_image})
                
                # convert image to bytes to save to DB (optional, but keep simple for now just save text)
                save_message("Image", "assistant", success_msg, {"agent": "Image"})
                
            elif agent_choice_logic == "Search":
                response = st.write_stream(search_stream(prompt, messages[:-1]))
                messages.append({"role": "assistant", "content": response, "agent": "Search"})
                save_message("Search", "assistant", response, {"agent": "Search"})

            elif agent_choice_logic == "RAG":
                if "vector_store" not in st.session_state:
                    response = "🐺: Please feed me intel first (upload documents in sidebar)."
                    st.markdown(response)
                    messages.append({"role": "assistant", "content": response, "agent": "RAG"})
                else:
                    # Get retrieved chunks for display
                    chunks = get_retrieved_chunks(prompt, st.session_state.vector_store)

                    response = st.write_stream(
                        ask_stream(prompt, st.session_state.vector_store, messages[:-1])
                    )

                    # Show retrieved chunks in expander
                    with st.expander("📂 Source Intelligence Fragments"):
                        render_chunks(chunks)

                    messages.append({
                        "role": "assistant",
                        "content": response,
                        "agent": "RAG",
                        "metadata": {"chunks": chunks}
                    })
                    save_message("RAG", "assistant", response, {"agent": "RAG", "chunks": chunks})

            elif agent_choice_logic == "Vision (OCR)":
                # Vision response is already handled in the sidebar for single uploads, 
                # but we add save logic if we were to allow follow-up questions
                pass

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("Rate limit reached. Please wait a moment and try again.")
            elif "api_key" in error_msg.lower() or "401" in error_msg:
                if agent_choice_logic == "Image":
                    st.error("Invalid or missing Hugging Face API key. Please check your HUGGINGFACE_API_KEY in the .env file.")
                else:
                    st.error("Invalid API key. Please check your GROQ_API_KEY in the .env file.")
            else:
                st.error(f"Something went wrong: {error_msg}")
