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

st.set_page_config(page_title="Agent Wolf", page_icon="🐺", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #1e1b4b 100%);
        color: #e2e8f0;
    }

    /* ---- Header ---- */
    .wolf-title {
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; letter-spacing: -1px; margin-bottom: 0.1rem;
    }
    .wolf-subtitle { color: #94a3b8; text-align: center; font-size: 1rem; margin-bottom: 0.15rem; }
    .wolf-credit {
        color: #6366f1; text-align: center; font-size: 0.78rem;
        margin-bottom: 1.2rem; letter-spacing: .08em;
        text-transform: uppercase; font-weight: 600;
    }

    /* ---- Mode Toolbar Pills ---- */
    .toolbar-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; align-items: center; }
    .mode-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 7px 14px; border-radius: 999px; font-size: 0.82rem;
        font-weight: 600; cursor: pointer; transition: all 0.2s ease;
        border: 1.5px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.04);
        color: #94a3b8; white-space: nowrap;
    }
    .mode-pill:hover { background: rgba(255,255,255,0.09); color: #e2e8f0; }

    .mode-pill.active-search  { background: rgba(99,102,241,0.2);  color: #818cf8; border-color: #6366f1; box-shadow: 0 0 12px rgba(99,102,241,0.35); }
    .mode-pill.active-arxiv   { background: rgba(14,165,233,0.2);  color: #38bdf8; border-color: #0ea5e9; box-shadow: 0 0 12px rgba(14,165,233,0.35); }
    .mode-pill.active-litrev  { background: rgba(168,85,247,0.2);  color: #c084fc; border-color: #a855f7; box-shadow: 0 0 12px rgba(168,85,247,0.35); }
    .mode-pill.active-gap     { background: rgba(239,68,68,0.2);   color: #f87171; border-color: #ef4444; box-shadow: 0 0 12px rgba(239,68,68,0.35); }
    .mode-pill.active-rag     { background: rgba(34,197,94,0.2);   color: #4ade80; border-color: #22c55e; box-shadow: 0 0 12px rgba(34,197,94,0.35); }
    .mode-pill.active-vision  { background: rgba(234,179,8,0.2);   color: #facc15; border-color: #eab308; box-shadow: 0 0 12px rgba(234,179,8,0.35); }
    .mode-pill.active-image   { background: rgba(236,72,153,0.2);  color: #f472b6; border-color: #ec4899; box-shadow: 0 0 12px rgba(236,72,153,0.35); }

    /* ---- Hide default Streamlit button styles for pills ---- */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 999px !important;
        padding: 6px 14px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: auto !important;
        white-space: nowrap !important;
    }

    /* ---- Chat Messages ---- */
    .stChatMessage {
        border-radius: 20px; border: 1px solid rgba(255,255,255,0.05);
        background: rgba(255,255,255,0.03); backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 1rem; padding: 1.5rem !important;
    }

    /* ---- Agent Badges ---- */
    .agent-badge {
        display: inline-flex; align-items: center;
        padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.1);
    }
    .badge-search  { background-color: rgba(99,102,241,0.2);  color: #818cf8; border-color: #6366f1; }
    .badge-rag     { background-color: rgba(34,197,94,0.2);   color: #4ade80; border-color: #22c55e; }
    .badge-vision  { background-color: rgba(234,179,8,0.2);   color: #facc15; border-color: #eab308; }
    .badge-image   { background-color: rgba(236,72,153,0.2);  color: #f472b6; border-color: #ec4899; }
    .badge-arxiv   { background-color: rgba(14,165,233,0.2);  color: #38bdf8; border-color: #0ea5e9; }
    .badge-litrev  { background-color: rgba(168,85,247,0.2);  color: #c084fc; border-color: #a855f7; }
    .badge-gap     { background-color: rgba(239,68,68,0.2);   color: #f87171; border-color: #ef4444; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: rgba(10,15,30,0.92);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    .sidebar-card {
        font-size: 0.85em; color: #94a3b8; padding: 12px;
        background: rgba(255,255,255,0.05); border-radius: 12px; text-align: center;
    }
    .info-card {
        border-radius: 12px; padding: 10px;
        font-size: 0.83em; color: #94a3b8; margin-bottom: 8px;
    }
    .info-card.blue  { background: rgba(14,165,233,0.08);  border: 1px solid rgba(14,165,233,0.25); }
    .info-card.purple{ background: rgba(168,85,247,0.08);  border: 1px solid rgba(168,85,247,0.25); }
    .info-card.red   { background: rgba(239,68,68,0.08);   border: 1px solid rgba(239,68,68,0.25); }
    .info-card.green { background: rgba(34,197,94,0.08);   border: 1px solid rgba(34,197,94,0.25); }
    .info-card.yellow{ background: rgba(234,179,8,0.08);   border: 1px solid rgba(234,179,8,0.25); }

    /* ---- Sidebar Buttons ---- */
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white; border: none; transition: all 0.3s ease; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99,102,241,0.4);
    }

    /* ---- Expander ---- */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important; background: rgba(255,255,255,0.02) !important;
    }

    /* ---- Footer ---- */
    .wolf-footer {
        text-align: center; color: #334155; font-size: 0.76rem;
        margin-top: 1.5rem; padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.04);
    }
    .wolf-footer span { color: #6366f1; }

    /* ---- Divider between toolbar and chat ---- */
    .toolbar-divider {
        height: 1px; background: rgba(255,255,255,0.06);
        margin: 6px 0 12px 0; border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  CONSTANTS
# ============================================================
MODES = [
    ("Search",    "🌐", "Global Search",    "badge-search",  "active-search"),
    ("ArXiv",     "🔬", "ArXiv Scholar",    "badge-arxiv",   "active-arxiv"),
    ("LitReview", "📖", "Lit Review",       "badge-litrev",  "active-litrev"),
    ("GapFinder", "🕵️", "Gap Finder",      "badge-gap",     "active-gap"),
    ("RAG",       "📚", "Knowledge Base",   "badge-rag",     "active-rag"),
    ("Vision",    "👁️", "Vision",           "badge-vision",  "active-vision"),
    ("Image",     "🎨", "Image Gen",        "badge-image",   "active-image"),
]

PLACEHOLDERS = {
    "Search":    "Search the web... e.g. 'Latest breakthroughs in AI 2025'",
    "ArXiv":     "Search papers... e.g. 'Transformer attention mechanisms'",
    "LitReview": "Generate a review... e.g. 'LLM hallucination detection'",
    "GapFinder": "Find research gaps in... e.g. 'Federated learning healthcare'",
    "RAG":       "Ask about your uploaded document...",
    "Vision":    "Upload an image via sidebar, then ask about it...",
    "Image":     "Generate an image... e.g. 'A wolf in a neon quantum lab'",
}


# ============================================================
#  SESSION STATE INIT
# ============================================================
init_db()

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Search"

for m, _, _, _, _ in MODES:
    key = f"msgs_{m}"
    if key not in st.session_state:
        st.session_state[key] = []

# Load DB history once per key
for m, _, _, _, _ in MODES:
    key = f"msgs_{m}"
    if not st.session_state[key]:
        st.session_state[key] = load_history(m)

# Persistent FAISS
if "vector_store" not in st.session_state and os.path.exists("database/faiss_index"):
    from utils.embeddings import SafeEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        st.session_state.vector_store = FAISS.load_local(
            "database/faiss_index", SafeEmbeddings(), allow_dangerous_deserialization=True
        )
    except Exception:
        pass


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-card">🐺 <b>Agent Wolf v2.0</b><br>'
        'Groq Llama 3.3 · ArXiv · FAISS<br>'
        '<span style="color:#6366f1;font-size:0.8em">Research Edition</span></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    mode = st.session_state.active_mode

    # Contextual info cards per mode
    if mode == "Search":
        st.markdown('<div class="info-card blue">🌐 <b>Global Search</b><br>Live web search via DuckDuckGo with source citations.</div>', unsafe_allow_html=True)
    elif mode == "ArXiv":
        st.markdown('<div class="info-card blue">🔬 <b>ArXiv Scholar</b><br>Search 2M+ real papers — get abstracts, PDF links, and AI synthesis.</div>', unsafe_allow_html=True)
    elif mode == "LitReview":
        st.markdown('<div class="info-card purple">📖 <b>Literature Review</b><br>Generates a full 5-section academic review using ArXiv + web sources.</div>', unsafe_allow_html=True)
    elif mode == "GapFinder":
        st.markdown('<div class="info-card red">🕵️ <b>Research Gap Finder</b><br>Identifies unexplored research areas and suggests your next paper.</div>', unsafe_allow_html=True)
    elif mode == "RAG":
        if "vector_store" in st.session_state:
            st.markdown('<div class="info-card green">📚 <b>Knowledge Base Ready</b><br>Documents processed. Ask anything about your files.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card green">📚 <b>Knowledge Base</b><br>Upload PDF or TXT files below to query your own documents.</div>', unsafe_allow_html=True)

        st.subheader("📥 Upload Sources")
        uploaded_files = st.file_uploader(
            "Feed documents to Wolf", type=["pdf", "txt"], accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("🚀 Process Documents"):
                with st.spinner("Building knowledge base..."):
                    try:
                        st.session_state.vector_store = build_vector_store(uploaded_files)
                        st.success(f"✅ {len(uploaded_files)} file(s) processed!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif mode == "Vision":
        st.markdown('<div class="info-card yellow">👁️ <b>Vision Analysis</b><br>Upload an image to extract text, describe content, or analyze charts.</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing..."):
                    try:
                        extracted = perform_ocr(uploaded_image.read())
                        st.session_state[f"msgs_Vision"].append({"role": "user", "content": f"📷 {uploaded_image.name}"})
                        st.session_state[f"msgs_Vision"].append({"role": "assistant", "content": extracted, "agent": "Vision"})
                        save_message("Vision", "user", f"📷 {uploaded_image.name}")
                        save_message("Vision", "assistant", extracted, {"agent": "Vision"})
                        st.success("Done!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    elif mode == "Image":
        st.markdown('<div class="info-card" style="background:rgba(236,72,153,0.08);border:1px solid rgba(236,72,153,0.25)">🎨 <b>Image Generation</b><br>Describe any image and Wolf will paint it with AI.</div>', unsafe_allow_html=True)

    st.divider()

    # Wipe session
    if st.button("🗑️ Wipe Session"):
        for m, _, _, _, _ in MODES:
            st.session_state[f"msgs_{m}"] = []
            clear_agent_history(m)
        st.rerun()

    # Export
    current_msgs = st.session_state.get(f"msgs_{mode}", [])
    if current_msgs:
        export_text = ""
        for msg in current_msgs:
            role = "You" if msg["role"] == "user" else f"Wolf ({msg.get('agent', mode)})"
            export_text += f"**{role}:**\n{msg['content']}\n\n---\n\n"
        st.download_button("💾 Export Chat", data=export_text,
                           file_name=f"wolf_{mode.lower()}_chat.md", mime="text/markdown")


# ============================================================
#  MAIN AREA — HEADER
# ============================================================
st.markdown('<div class="wolf-title">🐺 Agent Wolf</div>', unsafe_allow_html=True)
st.markdown('<div class="wolf-subtitle">Elite AI Research Copilot — Web · ArXiv · Vision · Knowledge</div>', unsafe_allow_html=True)
st.markdown('<div class="wolf-credit">✦ Deployed by Abdullah Ibne Tayeb Tamur ✦</div>', unsafe_allow_html=True)


def render_chunks(chunks):
    for i, chunk in enumerate(chunks, 1):
        relevance = chunk.get("relevance", 0)
        score     = chunk.get("score", 0)
        source    = chunk.get("source", "Unknown")
        page      = chunk.get("page")
        page_info = f" | Page {page + 1}" if page is not None else ""
        color_cls = "relevance-high" if relevance >= 60 else ("relevance-mid" if relevance >= 40 else "relevance-low")
        st.markdown(
            f'<div class="chunk-header"><strong>Chunk {i}</strong>'
            f'<span class="chunk-meta"><span class="{color_cls}">Relevance: {relevance}%</span>'
            f' | Score: {score:.3f}{page_info} | {source}</span></div>',
            unsafe_allow_html=True,
        )
        st.code(chunk["content"], language=None)
        if i < len(chunks):
            st.divider()


# ============================================================
#  CHAT HISTORY
# ============================================================
mode = st.session_state.active_mode
messages = st.session_state[f"msgs_{mode}"]

badge_map = {
    "Search": "badge-search", "ArXiv": "badge-arxiv", "LitReview": "badge-litrev",
    "GapFinder": "badge-gap", "RAG": "badge-rag", "Vision": "badge-vision", "Image": "badge-image",
}
label_map = {
    "Search": "🌐 Global Search", "ArXiv": "🔬 ArXiv Scholar", "LitReview": "📖 Lit Review",
    "GapFinder": "🕵️ Gap Finder", "RAG": "📚 Knowledge Base", "Vision": "👁️ Vision", "Image": "🎨 Image Gen",
}

for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            bc  = badge_map.get(message.get("agent", mode), "badge-search")
            lbl = label_map.get(message.get("agent", mode), "🐺 Wolf")
            st.markdown(f'<span class="agent-badge {bc}">{lbl}</span>', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"])
        if "metadata" in message and message["metadata"].get("chunks"):
            with st.expander("📂 Source Fragments"):
                render_chunks(message["metadata"]["chunks"])


# ============================================================
#  GEMINI-STYLE TOOLBAR — Mode Pills above chat input
# ============================================================
st.markdown('<div class="toolbar-divider"></div>', unsafe_allow_html=True)

# Row of pill buttons
cols = st.columns(len(MODES))
for i, (m_key, emoji, label, _, active_cls) in enumerate(MODES):
    is_active = (st.session_state.active_mode == m_key)
    with cols[i]:
        # Style the button differently based on active state
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{emoji} {label}", key=f"mode_btn_{m_key}", type=btn_type, use_container_width=True):
            st.session_state.active_mode = m_key
            st.rerun()


# ============================================================
#  CHAT INPUT
# ============================================================
mode = st.session_state.active_mode
messages = st.session_state[f"msgs_{mode}"]
placeholder = PLACEHOLDERS.get(mode, "Command Agent Wolf...")

if prompt := st.chat_input(placeholder):
    messages.append({"role": "user", "content": prompt})
    save_message(mode, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        bc  = badge_map.get(mode, "badge-search")
        lbl = label_map.get(mode, "🐺 Wolf")
        st.markdown(f'<span class="agent-badge {bc}">{lbl}</span>', unsafe_allow_html=True)

        try:
            if mode == "Search":
                response = st.write_stream(search_stream(prompt, messages[:-1]))
                messages.append({"role": "assistant", "content": response, "agent": "Search"})
                save_message("Search", "assistant", response, {"agent": "Search"})

            elif mode == "ArXiv":
                response = st.write_stream(arxiv_scholar_stream(prompt))
                messages.append({"role": "assistant", "content": response, "agent": "ArXiv"})
                save_message("ArXiv", "assistant", response, {"agent": "ArXiv"})

            elif mode == "LitReview":
                response = st.write_stream(literature_review_stream(prompt))
                messages.append({"role": "assistant", "content": response, "agent": "LitReview"})
                save_message("LitReview", "assistant", response, {"agent": "LitReview"})

            elif mode == "GapFinder":
                response = st.write_stream(research_gap_stream(prompt))
                messages.append({"role": "assistant", "content": response, "agent": "GapFinder"})
                save_message("GapFinder", "assistant", response, {"agent": "GapFinder"})

            elif mode == "RAG":
                if "vector_store" not in st.session_state:
                    response = "🐺 Please upload and process documents in the sidebar first."
                    st.markdown(response)
                    messages.append({"role": "assistant", "content": response, "agent": "RAG"})
                else:
                    chunks = get_retrieved_chunks(prompt, st.session_state.vector_store)
                    response = st.write_stream(ask_stream(prompt, st.session_state.vector_store, messages[:-1]))
                    with st.expander("📂 Source Fragments"):
                        render_chunks(chunks)
                    messages.append({"role": "assistant", "content": response, "agent": "RAG", "metadata": {"chunks": chunks}})
                    save_message("RAG", "assistant", response, {"agent": "RAG", "chunks": chunks})

            elif mode == "Image":
                gen_image = generate_image(prompt)
                st.image(gen_image)
                success_msg = f"🎨 Generated: *{prompt}*"
                messages.append({"role": "assistant", "content": success_msg, "agent": "Image", "image": gen_image})
                save_message("Image", "assistant", success_msg, {"agent": "Image"})

            elif mode == "Vision":
                pass  # handled in sidebar

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                st.error("⏳ Rate limit hit. Wait a moment and try again.")
            elif "api_key" in err.lower() or "401" in err:
                st.error("🔑 Invalid API key. Check your `.env` file.")
            else:
                st.error(f"Something went wrong: {err}")


# ============================================================
#  FOOTER
# ============================================================
st.markdown(
    '<div class="wolf-footer">🐺 <b>Agent Wolf v2.0</b> — AI Research Copilot<br>'
    'Powered by <span>Groq Llama 3.3</span> · ArXiv · FAISS · DuckDuckGo<br>'
    'Deployed by <span>Abdullah Ibne Tayeb Tamur</span></div>',
    unsafe_allow_html=True,
)
