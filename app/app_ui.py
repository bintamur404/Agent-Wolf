import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.search_agent import search_stream
from agents.rag_agent import build_vector_store, ask_stream, get_retrieved_chunks
from agents.ocr_agent import perform_ocr
from agents.image_agent import generate_image
from agents.arxiv_agent import arxiv_scholar_stream
from agents.research_agent import literature_review_stream, research_gap_stream
from utils.db import init_db, save_message, load_history, clear_agent_history

# ---- Page Config ----
st.set_page_config(page_title="Agent Wolf", page_icon="🐺", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #1e1b4b 100%);
        color: #e2e8f0;
    }

    /* Header */
    .wolf-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: -1px;
    }
    .wolf-subtitle {
        color: #94a3b8;
        text-align: center;
        margin-bottom: 0.2rem;
        font-size: 1.05rem;
    }
    .wolf-credit {
        color: #6366f1;
        text-align: center;
        font-size: 0.8rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Footer */
    .wolf-footer {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .wolf-footer span { color: #6366f1; }

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
    .badge-search   { background-color: rgba(99,102,241,0.2);  color: #818cf8; border-color: #6366f1; }
    .badge-rag      { background-color: rgba(34,197,94,0.2);   color: #4ade80; border-color: #22c55e; }
    .badge-vision   { background-color: rgba(234,179,8,0.2);   color: #facc15; border-color: #eab308; }
    .badge-image    { background-color: rgba(236,72,153,0.2);  color: #f472b6; border-color: #ec4899; }
    .badge-arxiv    { background-color: rgba(14,165,233,0.2);  color: #38bdf8; border-color: #0ea5e9; }
    .badge-litrev   { background-color: rgba(168,85,247,0.2);  color: #c084fc; border-color: #a855f7; }
    .badge-gap      { background-color: rgba(239,68,68,0.2);   color: #f87171; border-color: #ef4444; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.9);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-info {
        font-size: 0.85em;
        color: #94a3b8;
        padding: 12px;
        background: rgba(255,255,255,0.05);
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
        box-shadow: 0 10px 15px -3px rgba(99,102,241,0.4);
    }

    /* Expander */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* Research mode glow */
    .research-glow {
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 12px;
        padding: 10px;
        background: rgba(14,165,233,0.05);
        margin-bottom: 8px;
        font-size: 0.85em;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="wolf-title">🐺 Agent Wolf</div>', unsafe_allow_html=True)
st.markdown('<div class="wolf-subtitle">Elite AI Research Copilot — Web · ArXiv · Vision · Knowledge</div>', unsafe_allow_html=True)
st.markdown('<div class="wolf-credit">✦ Deployed by Abdullah Ibne Tayeb Tamur ✦</div>', unsafe_allow_html=True)


def render_chunks(chunks):
    for i, chunk in enumerate(chunks, 1):
        relevance = chunk.get("relevance", 0)
        score = chunk.get("score", 0)
        source = chunk.get("source", "Unknown")
        page = chunk.get("page")
        color_class = "relevance-high" if relevance >= 60 else ("relevance-mid" if relevance >= 40 else "relevance-low")
        page_info = f" | Page {page + 1}" if page is not None else ""
        st.markdown(
            f'<div class="chunk-header"><strong>Chunk {i}</strong>'
            f'<span class="chunk-meta"><span class="{color_class}">Relevance: {relevance}%</span>'
            f' | Distance: {score:.3f}{page_info} | Source: {source}</span></div>',
            unsafe_allow_html=True,
        )
        st.code(chunk["content"], language=None)
        if i < len(chunks):
            st.divider()


# ---- Sidebar ----
with st.sidebar:
    st.header("🎚️ Control Center")
    st.markdown(
        '<p class="sidebar-info">🐺 Agent Wolf v2.0<br>'
        'Groq Llama 3.3 · ArXiv · FAISS<br>'
        '<small style="color:#6366f1">Research Edition</small></p>',
        unsafe_allow_html=True,
    )

    agent_choice = st.radio(
        "Select Capability:",
        [
            "🌐 Global Search",
            "🔬 ArXiv Scholar",
            "📖 Literature Review",
            "🕵️ Research Gap Finder",
            "📚 Knowledge Base (RAG)",
            "👁️ Vision Analysis",
            "🎨 Image Generation",
        ],
        index=0,
    )

    agent_logic_map = {
        "🌐 Global Search":         "Search",
        "🔬 ArXiv Scholar":         "ArXiv",
        "📖 Literature Review":     "LitReview",
        "🕵️ Research Gap Finder":  "GapFinder",
        "📚 Knowledge Base (RAG)":  "RAG",
        "👁️ Vision Analysis":      "Vision (OCR)",
        "🎨 Image Generation":      "Image",
    }
    agent_choice_logic = agent_logic_map[agent_choice]

    st.divider()

    # Contextual info card per agent
    if agent_choice_logic == "Search":
        st.info("🌐 Wolf is scanning the World Wide Web via DuckDuckGo.")
    elif agent_choice_logic == "ArXiv":
        st.markdown(
            '<div class="research-glow">🔬 <b>ArXiv Scholar Active</b><br>'
            'Searches 2M+ real academic papers. Returns titles, abstracts, '
            'PDF links, and a synthesized analysis.</div>',
            unsafe_allow_html=True,
        )
    elif agent_choice_logic == "LitReview":
        st.markdown(
            '<div class="research-glow">📖 <b>Literature Review Mode</b><br>'
            'Combines ArXiv + web search to write a structured 5-section '
            'academic review of any topic.</div>',
            unsafe_allow_html=True,
        )
    elif agent_choice_logic == "GapFinder":
        st.markdown(
            '<div class="research-glow">🕵️ <b>Research Gap Finder</b><br>'
            'Analyzes the frontier of a field and pinpoints where the next '
            'breakthrough research should go.</div>',
            unsafe_allow_html=True,
        )
    elif agent_choice_logic == "RAG":
        if "vector_store" in st.session_state:
            st.success("📚 Knowledge base ready.")
        else:
            st.warning("⚠️ Upload documents below first.")
    elif agent_choice_logic == "Vision (OCR)":
        st.info("👁️ Vision sensors active.")
    else:
        st.info("🎨 Image generation core online.")

    st.divider()

    # RAG upload
    if agent_choice_logic == "RAG":
        st.subheader("📥 Upload Sources")
        uploaded_files = st.file_uploader(
            "Feed documents to Wolf", type=["pdf", "txt"], accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("🚀 Process Intelligence"):
                with st.spinner("Building knowledge base..."):
                    try:
                        st.session_state.vector_store = build_vector_store(uploaded_files)
                        st.success(f"Wolf assimilated {len(uploaded_files)} source(s)! ✅")
                    except Exception as e:
                        st.error(f"Brain malfunction: {e}")

    # Vision upload
    if agent_choice_logic == "Vision (OCR)":
        st.subheader("📷 Image Intel")
        uploaded_image = st.file_uploader("Show Wolf an image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            if st.button("🔍 Extract Vision"):
                with st.spinner("Analyzing image..."):
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

    # Wipe Session
    if st.button("🗑️ Wipe Session"):
        for key in ["search_messages", "rag_messages", "vision_messages",
                    "image_messages", "arxiv_messages", "litreview_messages", "gap_messages"]:
            st.session_state[key] = []
        for agent in ["Search", "RAG", "Vision", "Image", "ArXiv", "LitReview", "GapFinder"]:
            clear_agent_history(agent)
        st.rerun()

    # Export
    key_map = {
        "Search": "search_messages",
        "ArXiv": "arxiv_messages",
        "LitReview": "litreview_messages",
        "GapFinder": "gap_messages",
        "RAG": "rag_messages",
        "Vision (OCR)": "vision_messages",
        "Image": "image_messages",
    }
    current_key = key_map.get(agent_choice_logic, "search_messages")
    if current_key in st.session_state and st.session_state[current_key]:
        chat_export = ""
        for msg in st.session_state[current_key]:
            role = "You" if msg["role"] == "user" else f"Agent Wolf ({msg.get('agent', agent_choice_logic)})"
            chat_export += f"**{role}:**\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "💾 Export Session",
            data=chat_export,
            file_name=f"wolf_{agent_choice_logic.lower()}_session.md",
            mime="text/markdown",
        )

# ---- Session State Init ----
init_db()
for key in ["search_messages", "rag_messages", "vision_messages",
            "image_messages", "arxiv_messages", "litreview_messages", "gap_messages"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Load DB history only once
if not st.session_state.search_messages:
    st.session_state.search_messages   = load_history("Search")
if not st.session_state.rag_messages:
    st.session_state.rag_messages      = load_history("RAG")
if not st.session_state.vision_messages:
    st.session_state.vision_messages   = load_history("Vision")
if not st.session_state.image_messages:
    st.session_state.image_messages    = load_history("Image")
if not st.session_state.arxiv_messages:
    st.session_state.arxiv_messages    = load_history("ArXiv")
if not st.session_state.litreview_messages:
    st.session_state.litreview_messages = load_history("LitReview")
if not st.session_state.gap_messages:
    st.session_state.gap_messages      = load_history("GapFinder")

# Persistent FAISS Index
if "vector_store" not in st.session_state and os.path.exists("database/faiss_index"):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        st.session_state.vector_store = FAISS.load_local(
            "database/faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        pass

# ---- Active Messages ----
messages_map = {
    "Search":      st.session_state.search_messages,
    "ArXiv":       st.session_state.arxiv_messages,
    "LitReview":   st.session_state.litreview_messages,
    "GapFinder":   st.session_state.gap_messages,
    "RAG":         st.session_state.rag_messages,
    "Vision (OCR)":st.session_state.vision_messages,
    "Image":       st.session_state.image_messages,
}
messages = messages_map[agent_choice_logic]

badge_map = {
    "Search": "badge-search", "ArXiv": "badge-arxiv",
    "LitReview": "badge-litrev", "GapFinder": "badge-gap",
    "RAG": "badge-rag", "Vision (OCR)": "badge-vision", "Image": "badge-image",
}
label_map = {
    "Search": "🌐 Global Search", "ArXiv": "🔬 ArXiv Scholar",
    "LitReview": "📖 Literature Review", "GapFinder": "🕵️ Gap Finder",
    "RAG": "📚 Knowledge Base", "Vision (OCR)": "👁️ Vision", "Image": "🎨 Image Gen",
}

# ---- Placeholder hint per mode ----
placeholder_map = {
    "Search":       "Search the web... e.g. 'Latest AI breakthroughs 2025'",
    "ArXiv":        "Search academic papers... e.g. 'Transformer attention mechanisms'",
    "LitReview":    "Generate a literature review... e.g. 'LLM hallucination detection'",
    "GapFinder":    "Find research gaps in... e.g. 'Federated learning for healthcare'",
    "RAG":          "Ask about your uploaded documents...",
    "Vision (OCR)": "Use the sidebar to upload and analyze an image...",
    "Image":        "Describe an image to generate... e.g. 'A wolf in a quantum lab'",
}

# ---- Display Chat History ----
for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            bc = badge_map.get(message.get("agent", agent_choice_logic), "badge-search")
            lbl = label_map.get(message.get("agent", agent_choice_logic), "🐺 Wolf")
            st.markdown(f'<span class="agent-badge {bc}">{lbl}</span>', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"])
        if "metadata" in message and message["metadata"].get("chunks"):
            with st.expander("📂 Source Intelligence Fragments"):
                render_chunks(message["metadata"]["chunks"])

# ---- Handle User Input ----
if prompt := st.chat_input(placeholder_map.get(agent_choice_logic, "Command Agent Wolf...")):
    messages.append({"role": "user", "content": prompt})
    save_message(agent_choice_logic, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        bc = badge_map.get(agent_choice_logic, "badge-search")
        lbl = label_map.get(agent_choice_logic, "🐺 Wolf")
        st.markdown(f'<span class="agent-badge {bc}">{lbl}</span>', unsafe_allow_html=True)

        try:
            # --- Global Search ---
            if agent_choice_logic == "Search":
                response = st.write_stream(search_stream(prompt, messages[:-1]))
                messages.append({"role": "assistant", "content": response, "agent": "Search"})
                save_message("Search", "assistant", response, {"agent": "Search"})

            # --- ArXiv Scholar ---
            elif agent_choice_logic == "ArXiv":
                response = st.write_stream(arxiv_scholar_stream(prompt))
                messages.append({"role": "assistant", "content": response, "agent": "ArXiv"})
                save_message("ArXiv", "assistant", response, {"agent": "ArXiv"})

            # --- Literature Review ---
            elif agent_choice_logic == "LitReview":
                response = st.write_stream(literature_review_stream(prompt))
                messages.append({"role": "assistant", "content": response, "agent": "LitReview"})
                save_message("LitReview", "assistant", response, {"agent": "LitReview"})

            # --- Research Gap Finder ---
            elif agent_choice_logic == "GapFinder":
                response = st.write_stream(research_gap_stream(prompt))
                messages.append({"role": "assistant", "content": response, "agent": "GapFinder"})
                save_message("GapFinder", "assistant", response, {"agent": "GapFinder"})

            # --- RAG ---
            elif agent_choice_logic == "RAG":
                if "vector_store" not in st.session_state:
                    response = "🐺 Please upload and process documents in the sidebar first."
                    st.markdown(response)
                    messages.append({"role": "assistant", "content": response, "agent": "RAG"})
                else:
                    chunks = get_retrieved_chunks(prompt, st.session_state.vector_store)
                    response = st.write_stream(ask_stream(prompt, st.session_state.vector_store, messages[:-1]))
                    with st.expander("📂 Source Intelligence Fragments"):
                        render_chunks(chunks)
                    messages.append({"role": "assistant", "content": response, "agent": "RAG", "metadata": {"chunks": chunks}})
                    save_message("RAG", "assistant", response, {"agent": "RAG", "chunks": chunks})

            # --- Image Generation ---
            elif agent_choice_logic == "Image":
                gen_image = generate_image(prompt)
                st.image(gen_image)
                success_msg = f"Painted: {prompt}"
                messages.append({"role": "assistant", "content": success_msg, "agent": "Image", "image": gen_image})
                save_message("Image", "assistant", success_msg, {"agent": "Image"})

            # --- Vision ---
            elif agent_choice_logic == "Vision (OCR)":
                pass  # handled in sidebar

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("⏳ Rate limit reached. Please wait a moment and try again.")
            elif "api_key" in error_msg.lower() or "401" in error_msg:
                st.error("🔑 Invalid API key. Check your `.env` file.")
            else:
                st.error(f"Something went wrong: {error_msg}")

# ---- Footer ----
st.markdown(
    '<div class="wolf-footer">🐺 <b>Agent Wolf v2.0</b> — AI Research Copilot<br>'
    'Powered by <span>Groq Llama 3.3</span> · ArXiv · FAISS · DuckDuckGo<br>'
    'Deployed by <span>Abdullah Ibne Tayeb Tamur</span></div>',
    unsafe_allow_html=True,
)
