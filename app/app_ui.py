import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.search_agent import search_stream
from agents.rag_agent import build_vector_store, ask_stream, get_retrieved_chunks
from agents.ocr_agent import perform_ocr
from agents.image_agent import generate_image
from agents.arxiv_agent import search_arxiv, format_papers_for_llm
from agents.research_agent import literature_review_stream, research_gap_stream
from utils.db import init_db, save_message, load_history, clear_agent_history
from utils.llm import get_llm
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(
    page_title="Agent Wolf",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
#  GEMINI-STYLE CSS — Clean, minimal dark. NOT cyberpunk.
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Inter:wght@300;400;500;600&display=swap');

    /* ---- Base ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Google Sans', sans-serif;
    }
    .stApp {
        background-color: #131314;
        color: #e8eaed;
    }

    /* ---- Hide Streamlit chrome ---- */
    #MainMenu, footer, .stDeployButton { display: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ---- Main container width constraint ---- */
    .block-container {
        max-width: 860px !important;
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        margin: 0 auto !important;
    }

    /* ---- Header ---- */
    .gw-header {
        text-align: center;
        padding: 2.4rem 0 0.5rem 0;
    }
    .gw-logo {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4285f4, #9b59b6, #ea4335);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .gw-tagline {
        font-size: 0.9rem;
        color: #9aa0a6;
        margin-top: 0.2rem;
        font-weight: 400;
    }
    .gw-credit {
        font-size: 0.72rem;
        color: #5f6368;
        margin-top: 0.25rem;
        letter-spacing: 0.04em;
    }

    /* ---- Mode Pill Bar ---- */
    .gw-pill-bar {
        display: flex;
        gap: 6px;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1rem 0 0.6rem 0;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 999px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        border: 1px solid #3c4043 !important;
        background: #1e1f20 !important;
        color: #9aa0a6 !important;
        transition: all 0.18s ease !important;
        white-space: nowrap !important;
        width: auto !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #2a2b2d !important;
        color: #e8eaed !important;
        border-color: #5f6368 !important;
    }
    /* Primary (active) pill */
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background: #1a3a5c !important;
        color: #8ab4f8 !important;
        border-color: #4285f4 !important;
    }

    /* ---- Chat Messages ---- */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.6rem 0 !important;
        margin-bottom: 0.2rem !important;
    }
    /* User bubble */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: #1e1f20 !important;
        border-radius: 18px !important;
        padding: 0.8rem 1.2rem !important;
        max-width: 80%;
        margin-left: auto !important;
    }
    /* Assistant area */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        padding: 0.6rem 0 !important;
    }

    /* ---- Agent label pill ---- */
    .agent-label {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.75rem; font-weight: 600;
        color: #8ab4f8; margin-bottom: 6px;
        letter-spacing: 0.03em;
    }
    .agent-label.lab-arxiv  { color: #81c995; }
    .agent-label.lab-review { color: #c58af9; }
    .agent-label.lab-gap    { color: #f28b82; }
    .agent-label.lab-rag    { color: #81c995; }
    .agent-label.lab-vision { color: #fdd663; }
    .agent-label.lab-image  { color: #f28b82; }

    /* ---- Popover / Expander ---- */
    .stPopover, .stExpander {
        border: 1px solid #3c4043 !important;
        background: #1e1f20 !important;
        border-radius: 12px !important;
    }

    /* ---- Chat input ---- */
    .stChatInputContainer, div[data-testid="stChatInput"] > div {
        background: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 24px !important;
    }
    .stChatInputContainer:focus-within, div[data-testid="stChatInput"] > div:focus-within {
        border-color: #4285f4 !important;
        box-shadow: 0 0 0 2px rgba(66,133,244,0.18) !important;
    }

    /* ---- Generic button (non-pill) ---- */
    .stButton > button {
        border-radius: 8px;
        background: #1e1f20;
        color: #e8eaed;
        border: 1px solid #3c4043;
        transition: all 0.15s;
    }
    .stButton > button:hover {
        background: #2a2b2d;
        border-color: #5f6368;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: #1e1f20 !important;
        border: 1px dashed #3c4043 !important;
        border-radius: 12px !important;
    }

    /* ---- Attachment badge ---- */
    .att-badge {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 0.78rem; color: #81c995;
        background: rgba(129,201,149,0.1);
        border: 1px solid rgba(129,201,149,0.3);
        border-radius: 999px; padding: 4px 12px;
        margin-bottom: 6px;
    }
    .att-badge-img {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 0.78rem; color: #fdd663;
        background: rgba(253,214,99,0.1);
        border: 1px solid rgba(253,214,99,0.3);
        border-radius: 999px; padding: 4px 12px;
        margin-bottom: 6px;
    }

    /* ---- Status/thinking block ---- */
    [data-testid="stStatus"], [data-testid="stStatusWidget"] {
        background: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 10px !important;
        font-size: 0.83rem !important;
        color: #9aa0a6 !important;
    }

    /* ---- Source expander ---- */
    .src-expander {
        font-size: 0.78rem;
        color: #9aa0a6;
        margin-top: 8px;
    }

    /* ---- Divider ---- */
    hr { border-color: #2a2b2d !important; }

    /* ---- Footer ---- */
    .gw-footer {
        text-align: center;
        font-size: 0.72rem;
        color: #3c4043;
        margin-top: 2rem;
        padding-top: 0.8rem;
        border-top: 1px solid #2a2b2d;
    }
    .gw-footer a { color: #5f6368; text-decoration: none; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #1e1f20 !important;
        border-right: 1px solid #2a2b2d !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  CONSTANTS
# ============================================================
MODES = [
    ("Search",    "🌐", "Web Search"),
    ("ArXiv",     "🔬", "ArXiv Scholar"),
    ("LitReview", "📖", "Lit Review"),
    ("GapFinder", "🕵️", "Gap Finder"),
    ("RAG",       "📚", "Knowledge Base"),
    ("Vision",    "👁️",  "Vision"),
    ("Image",     "🎨", "Image Gen"),
]

LABEL_MAP = {
    "Search":    "🌐 Web Search",
    "ArXiv":     "🔬 ArXiv Scholar",
    "LitReview": "📖 Lit Review",
    "GapFinder": "🕵️ Gap Finder",
    "RAG":       "📚 Knowledge Base",
    "Vision":    "👁️ Vision",
    "Image":     "🎨 Image Gen",
}

LABEL_CLASS = {
    "Search": "", "ArXiv": "lab-arxiv", "LitReview": "lab-review",
    "GapFinder": "lab-gap", "RAG": "lab-rag", "Vision": "lab-vision", "Image": "lab-image",
}

HINTS = {
    "Search":    "Ask anything — Wolf searches the web for you...",
    "ArXiv":     "Search academic papers... e.g. 'attention mechanisms 2024'",
    "LitReview": "Generate a lit review on... e.g. 'LLM hallucination'",
    "GapFinder": "Find research gaps in... e.g. 'federated learning'",
    "RAG":       "Ask about your document...",
    "Vision":    "Attach an image via ➕, then ask about it here...",
    "Image":     "Describe what to generate... e.g. 'A wolf in a Tokyo lab'",
}

# ============================================================
#  SESSION STATE
# ============================================================
init_db()

_defaults = {
    "active_mode":    "Search",
    "attached_pdf":   None,
    "attached_image": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

for m, _, _ in MODES:
    key = f"msgs_{m}"
    if key not in st.session_state:
        st.session_state[key] = []

for m, _, _ in MODES:
    key = f"msgs_{m}"
    if not st.session_state[key]:
        st.session_state[key] = load_history(m)

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
#  SIDEBAR — minimal
# ============================================================
with st.sidebar:
    st.markdown("### 🐺 Agent Wolf v2.0")
    st.caption("Research Edition · Groq Llama 3.3")
    st.divider()

    mode_sb = st.session_state.active_mode
    if mode_sb == "RAG":
        if "vector_store" in st.session_state:
            st.success("📚 Knowledge base ready")
        else:
            st.info("Upload a PDF via the ➕ attach button in chat")
    elif mode_sb == "Vision" and st.session_state.attached_image:
        st.success(f"🖼️ {st.session_state.attached_image['name']}")

    st.divider()
    if st.button("🗑️ Clear session"):
        for m, _, _ in MODES:
            st.session_state[f"msgs_{m}"] = []
            clear_agent_history(m)
        st.session_state.attached_pdf   = None
        st.session_state.attached_image = None
        st.rerun()

    _cur = st.session_state.get(f"msgs_{mode_sb}", [])
    if _cur:
        _exp = "\n\n".join(
            f"**{'You' if m['role']=='user' else 'Wolf'}:** {m['content']}"
            for m in _cur
        )
        st.download_button("💾 Export chat", data=_exp,
                           file_name=f"wolf_{mode_sb}.md", mime="text/markdown")
    st.divider()
    st.caption("Engineered by **Abdullah Ibne Tayeb Tamur**")

# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="gw-header">
  <div class="gw-logo">🐺 Agent Wolf</div>
  <div class="gw-tagline">Multi-Modal AI Research Copilot</div>
  <div class="gw-credit">Deployed by Abdullah Ibne Tayeb Tamur</div>
</div>
""", unsafe_allow_html=True)




# ============================================================
#  HELPERS
# ============================================================
def render_chunks(chunks):
    for i, c in enumerate(chunks, 1):
        rel   = c.get("relevance", 0)
        score = c.get("score", 0)
        src   = c.get("source", "?")
        pg    = c.get("page")
        pg_s  = f" · p.{pg+1}" if pg is not None else ""
        st.markdown(f"**Chunk {i}** — `{rel}%` relevance · dist `{score:.3f}`{pg_s} · *{src}*")
        st.code(c["content"], language=None)
        if i < len(chunks):
            st.divider()


def agent_label(mode: str):
    lbl = LABEL_MAP.get(mode, mode)
    cls = LABEL_CLASS.get(mode, "")
    st.markdown(f'<div class="agent-label {cls}">{lbl}</div>', unsafe_allow_html=True)


# ============================================================
#  CHAT HISTORY
# ============================================================
mode     = st.session_state.active_mode
messages = st.session_state[f"msgs_{mode}"]

for msg in messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            agent_label(msg.get("agent", mode))
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"])
        if "metadata" in msg and msg["metadata"].get("chunks"):
            with st.expander("📂 Source fragments", expanded=False):
                render_chunks(msg["metadata"]["chunks"])

# ============================================================
#  ATTACHMENT STATUS BADGE
# ============================================================
if st.session_state.attached_pdf:
    st.markdown(
        f'<div class="att-badge">📄 {st.session_state.attached_pdf} &nbsp;·&nbsp; Knowledge Base ready</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.attached_image:
    st.markdown(
        f'<div class="att-badge-img">🖼️ {st.session_state.attached_image["name"]} &nbsp;·&nbsp; Vision ready</div>',
        unsafe_allow_html=True,
    )

# ============================================================
#  INPUT ROW:  [➕ popover]  [chat input]
# ============================================================

# Active mode indicator above input
_am = st.session_state.active_mode
_emoji = next((e for k, e, _ in MODES if k == _am), "🐺")
_lbl   = next((l for k, _, l in MODES if k == _am), _am)

col_plus, col_chat = st.columns([0.07, 0.93])

with col_plus:
    with st.popover("➕", help="Switch agent or attach a file"):

        # ── SECTION 1: Agent / Tool selector ──
        st.markdown("**🔧 Switch Agent**")
        st.caption("Choose what Wolf should do with your message")
        for m_key, m_emoji, m_label in MODES:
            is_active = (st.session_state.active_mode == m_key)
            label_txt = f"{m_emoji} {m_label}" + (" ✓" if is_active else "")
            if st.button(label_txt, key=f"pop_mode_{m_key}", use_container_width=True):
                st.session_state.active_mode = m_key
                st.rerun()

        st.divider()

        # ── SECTION 2: Attach Document ──
        st.markdown("**📄 Attach Document** *(PDF / TXT → Knowledge Base)*")
        doc_up = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"],
                                  key="pop_doc", label_visibility="collapsed")
        if doc_up and st.button("✅ Load document", key="load_pop_doc"):
            with st.spinner(f"Processing {doc_up.name}..."):
                try:
                    st.session_state.vector_store = build_vector_store([doc_up])
                    st.session_state.attached_pdf   = doc_up.name
                    st.session_state.attached_image = None
                    st.session_state.active_mode    = "RAG"
                    st.success("✅ Done — ask your question below!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        # ── SECTION 3: Attach Image ──
        st.markdown("**🖼️ Attach Image** *(PNG / JPG → Vision)*")
        img_up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"],
                                  key="pop_img", label_visibility="collapsed")
        if img_up:
            st.image(img_up, use_container_width=True)
        if img_up and st.button("✅ Analyze image", key="analyze_pop_img"):
                with st.spinner(f"Analyzing {img_up.name}..."):
                    try:
                        extracted = perform_ocr(img_up.read())
                        st.session_state.attached_image = {"name": img_up.name, "text": extracted}
                        st.session_state.attached_pdf   = None
                        st.session_state.active_mode    = "Vision"
                        st.session_state["msgs_Vision"].append(
                            {"role": "user", "content": f"📷 Attached: **{img_up.name}**"}
                        )
                        st.session_state["msgs_Vision"].append(
                            {"role": "assistant", "content": extracted, "agent": "Vision"}
                        )
                        save_message("Vision", "user",      f"📷 Attached: {img_up.name}")
                        save_message("Vision", "assistant", extracted, {"agent": "Vision"})
                        st.success("✅ Image analyzed — ask follow-up questions below!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.attached_pdf or st.session_state.attached_image:
            if st.button("🗑️ Remove attachment", key="rm_attach"):
                st.session_state.attached_pdf   = None
                st.session_state.attached_image = None
                st.rerun()

# ============================================================
#  CHAT INPUT
# ============================================================
with col_chat:
    mode     = st.session_state.active_mode
    messages = st.session_state[f"msgs_{mode}"]
    prompt   = st.chat_input(HINTS.get(mode, "Ask Wolf anything..."))

if prompt:
    messages.append({"role": "user", "content": prompt})
    save_message(mode, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        agent_label(mode)

        try:
            # ---- 🌐 Web Search ----
            if mode == "Search":
                with st.status("🌐 Searching the web...", expanded=True) as tool_status:
                    ddg   = DuckDuckGoSearchRun()
                    query = prompt
                    st.write(f"**Query:** `{query}`")
                    web_r = ddg.run(query)
                    tool_status.update(label="✅ Web search complete", state="complete", expanded=False)

                # Stream synthesis
                from langchain_core.messages import SystemMessage, HumanMessage
                llm  = get_llm()
                resp = st.write_stream(
                    c.content for c in llm.stream([
                        SystemMessage(content=(
                            "You are a concise web research assistant. Summarize the search results below "
                            "clearly. Cite sources if URLs appear in the results. Be direct and useful.\n\n"
                            f"Search results:\n{web_r}"
                        )),
                        HumanMessage(content=prompt),
                    ]) if c.content
                )
                messages.append({"role": "assistant", "content": resp, "agent": "Search"})
                save_message("Search", "assistant", resp, {"agent": "Search"})

            # ---- 🔬 ArXiv Scholar ----
            elif mode == "ArXiv":
                with st.status("🔬 Searching ArXiv database...", expanded=True) as tool_status:
                    st.write(f"**Query:** `{prompt}`")
                    papers = search_arxiv(prompt, max_results=5)
                    tool_status.update(label=f"✅ Found {len(papers)} papers", state="complete", expanded=False)

                if not papers:
                    st.warning("No papers found. Try broader terms.")
                else:
                    st.markdown("**📄 Papers found:**")
                    for i, p in enumerate(papers, 1):
                        with st.expander(f"[{i}] {p['title']} — {p['authors']} ({p['year']})", expanded=False):
                            st.caption(f"Categories: `{p['categories']}`")
                            st.write(p["abstract"])
                            c1, c2 = st.columns(2)
                            c1.markdown(f"[🔗 ArXiv]({p['url']})")
                            c2.markdown(f"[📥 PDF]({p['pdf_url']})")

                    from langchain_core.messages import SystemMessage, HumanMessage
                    llm = get_llm()
                    st.markdown("**🧠 Synthesis:**")
                    resp = st.write_stream(
                        c.content for c in llm.stream([
                            SystemMessage(content=(
                                "Synthesize the key findings from these ArXiv papers. "
                                "Cite each paper by [[Title]](url). End with 3-5 Key Takeaways."
                            )),
                            HumanMessage(content=(
                                f"Query: {prompt}\n\n"
                                f"Papers:\n{format_papers_for_llm(papers)}"
                            )),
                        ]) if c.content
                    )
                    messages.append({"role": "assistant", "content": resp, "agent": "ArXiv"})
                    save_message("ArXiv", "assistant", resp, {"agent": "ArXiv"})

            # ---- 📖 Literature Review ----
            elif mode == "LitReview":
                with st.status("📖 Gathering sources for literature review...", expanded=True) as ts:
                    st.write(f"**Topic:** `{prompt}`")
                    ddg    = DuckDuckGoSearchRun()
                    papers = search_arxiv(prompt, max_results=6)
                    web_r  = ddg.run(f"{prompt} research 2024 2025")
                    ts.update(label=f"✅ {len(papers)} papers + web context", state="complete", expanded=False)
                resp = st.write_stream(literature_review_stream(prompt))
                messages.append({"role": "assistant", "content": resp, "agent": "LitReview"})
                save_message("LitReview", "assistant", resp, {"agent": "LitReview"})

            # ---- 🕵️ Research Gap Finder ----
            elif mode == "GapFinder":
                with st.status("🕵️ Scanning research frontier...", expanded=True) as ts:
                    st.write(f"**Field:** `{prompt}`")
                    papers = search_arxiv(prompt, max_results=8)
                    ts.update(label=f"✅ Analyzed {len(papers)} frontier papers", state="complete", expanded=False)
                resp = st.write_stream(research_gap_stream(prompt))
                messages.append({"role": "assistant", "content": resp, "agent": "GapFinder"})
                save_message("GapFinder", "assistant", resp, {"agent": "GapFinder"})

            # ---- 📚 Knowledge Base (RAG) ----
            elif mode == "RAG":
                if "vector_store" not in st.session_state:
                    st.info("Click **➕** to upload a document first.")
                else:
                    with st.status("📚 Retrieving from knowledge base...", expanded=True) as ts:
                        st.write(f"**Query:** `{prompt}`")
                        chunks = get_retrieved_chunks(prompt, st.session_state.vector_store)
                        ts.update(label=f"✅ Retrieved {len(chunks)} relevant chunks", state="complete", expanded=False)
                    resp = st.write_stream(ask_stream(prompt, st.session_state.vector_store, messages[:-1]))
                    with st.expander("📂 Source fragments", expanded=False):
                        render_chunks(chunks)
                    messages.append({"role": "assistant", "content": resp, "agent": "RAG",
                                     "metadata": {"chunks": chunks}})
                    save_message("RAG", "assistant", resp, {"agent": "RAG", "chunks": chunks})

            # ---- 🎨 Image Generation ----
            elif mode == "Image":
                with st.status("🎨 Generating image...", expanded=True) as ts:
                    st.write(f"**Prompt:** `{prompt}`")
                    gen_img = generate_image(prompt)
                    ts.update(label="✅ Image generated", state="complete", expanded=False)
                st.image(gen_img)
                msg_ = f"🎨 Generated: *{prompt}*"
                messages.append({"role": "assistant", "content": msg_,
                                 "agent": "Image", "image": gen_img})
                save_message("Image", "assistant", msg_, {"agent": "Image"})

            # ---- 👁️ Vision (follow-up) ----
            elif mode == "Vision":
                if st.session_state.attached_image:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    ctx = st.session_state.attached_image["text"]
                    llm = get_llm()
                    with st.status("👁️ Reasoning over image content...", expanded=True) as ts:
                        st.write(f"**Question:** `{prompt}`")
                        ts.update(label="✅ Context retrieved", state="complete", expanded=False)
                    resp = st.write_stream(c.content for c in llm.stream([
                        SystemMessage(content=(
                            "You are an expert image analyst. Use the extracted image content below "
                            "to answer follow-up questions clearly and accurately.\n\n"
                            f"Extracted content:\n{ctx}"
                        )),
                        HumanMessage(content=prompt),
                    ]) if c.content)
                    messages.append({"role": "assistant", "content": resp, "agent": "Vision"})
                    save_message("Vision", "assistant", resp, {"agent": "Vision"})
                else:
                    st.info("Click **➕** to attach an image first.")

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                st.error("⏳ Rate limit reached — please wait a moment and retry.")
            elif "401" in err or "api_key" in err.lower():
                st.error("🔑 Invalid API key. Check `.env`.")
            else:
                st.error(f"Error: {err}")

# ============================================================
#  FOOTER
# ============================================================
st.markdown(
    '<div class="gw-footer">'
    '🐺 Agent Wolf v2.0 &nbsp;·&nbsp; Powered by Groq Llama 3.3, ArXiv, DuckDuckGo, FAISS<br>'
    'Deployed by <strong>Abdullah Ibne Tayeb Tamur</strong>'
    '</div>',
    unsafe_allow_html=True,
)
