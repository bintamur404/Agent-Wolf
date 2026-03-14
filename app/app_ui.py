import streamlit as st
import os
import sys
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.rag_agent import build_vector_store, ask_stream, get_retrieved_chunks
from agents.ocr_agent import perform_ocr
from agents.image_agent import generate_image
from agents.arxiv_agent import search_arxiv, format_papers_for_llm
from agents.research_agent import literature_review_stream, research_gap_stream
from utils.db import init_db, save_message, load_history, clear_agent_history
from utils.llm import get_llm
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(
    page_title="Wolf Scholar",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  PREMIUM CSS — Dark SaaS, not cyberpunk
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* === Base === */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* === Hide Streamlit chrome === */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }
    [data-testid="stHeader"] { height: 0px !important; }

    /* === App background === */
    .stApp { background: #111214; color: #e2e2e2; }

    /* === Constrain main content === */
    .block-container {
        max-width: 820px !important;
        padding-top: 1rem !important;
        padding-bottom: 10rem !important;
        margin: 0 auto !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }

    /* === Header === */
    .ws-header {
        padding: 2.5rem 0 1rem;
        text-align: center;
    }
    .ws-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .ws-sub {
        color: #6b7280; font-size: 0.88rem; margin-top: 4px;
    }

    /* === Chat messages === */
    .stChatMessage {
        border-radius: 15px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.6rem !important;
        border: 1px solid #23252a !important;
        background: #18191c !important;
    }
    [data-testid="stChatMessageUser"] {
        background: #1d1e23 !important;
        margin-left: 10% !important;
    }

    /* === Input area === */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        background: #1c1d21 !important;
        border: 1px solid #2e3038 !important;
        color: #e2e2e2 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.15) !important;
    }

    /* === File uploader === */
    [data-testid="stFileUploaderDropzone"] {
        background: #1c1d21 !important;
        border: 1px dashed #2e3038 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] { color: #6b7280 !important; }

    /* === Status widget === */
    [data-testid="stStatus"] {
        background: #1c1d21 !important;
        border: 1px solid #2e3038 !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
    }

    /* === Expander === */
    .stExpander {
        background: #1c1d21 !important;
        border: 1px solid #2e3038 !important;
        border-radius: 12px !important;
    }

    /* === Buttons === */
    .stButton > button {
        border-radius: 10px !important;
        background: #1c1d21 !important;
        color: #e2e2e2 !important;
        border: 1px solid #2e3038 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: #26272d !important;
        border-color: #60a5fa !important;
        color: #60a5fa !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
        color: white !important; border: none !important;
    }

    /* === Popover === */
    [data-testid="stPopover"] {
        background: #1c1d21 !important;
        border: 1px solid #2e3038 !important;
        border-radius: 14px !important;
    }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: #16171a !important;
        border-right: 1px solid #23252a !important;
    }
    section[data-testid="stSidebar"] * { color: #9ca3af; }

    /* === Active tool badge === */
    .ws-tool-badge {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.75rem; color: #60a5fa;
        background: rgba(96,165,250,0.1);
        border: 1px solid rgba(96,165,250,0.25);
        border-radius: 999px; padding: 3px 10px;
        margin-bottom: 8px;
    }
    .ws-attach-preview {
        background: #1c1d21; border: 1px solid #2e3038;
        border-radius: 10px; padding: 8px 12px;
        font-size: 0.8rem; color: #9ca3af; margin-bottom: 8px;
        display: flex; align-items: center; gap: 8px;
    }

    /* Divider */
    hr { border-color: #23252a !important; margin: 0.6rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  CONSTANTS & SESSION STATE
# ============================================================
ADVANCED_MODES = [
    ("ArXiv",     "🔬", "ArXiv Scholar",    "Deep dive into academic papers"),
    ("LitReview", "📖", "Literature Review", "Full structured academic review"),
    ("GapFinder", "🕵️", "Research Gap Finder","Find what hasn't been studied"),
    ("ImageGen",  "🎨", "Image Generation",  "Generate images from a prompt"),
]

init_db()

for k, v in {
    "messages":       [],
    "vector_store":   None,
    "active_mode":    "auto",   # "auto" = route by file. Others: arxiv, litreview, gap, image
    "pending_file":   None,     # {name, bytes, type}
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state["messages"]:
    st.session_state["messages"] = load_history("Wolf")

if st.session_state["vector_store"] is None and os.path.exists("database/faiss_index"):
    from utils.embeddings import SafeEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        st.session_state["vector_store"] = FAISS.load_local(
            "database/faiss_index", SafeEmbeddings(), allow_dangerous_deserialization=True
        )
    except Exception:
        pass


# ============================================================
#  SIDEBAR — Portfolio branding + session controls
# ============================================================
with st.sidebar:
    st.markdown("""
<div style='text-align: center; padding-top: 20px;'>
    <p style='color: #888; font-size: 14px;'>Engineered & Deployed by</p>
    <h3 style='color: #4CAF50; margin-top: -10px;'>Abdullah Ibne Tayeb Tamur</h3>
    <p style='color: #888; font-size: 12px;'>Multi-Modal AI Research Architecture</p>
</div>
    """, unsafe_allow_html=True)

    st.divider()

    # Active mode indicator
    mode_labels = {
        "auto":      "🤖 Auto-Route  (file-based)",
        "ArXiv":     "🔬 ArXiv Scholar",
        "LitReview": "📖 Literature Review",
        "GapFinder": "🕵️ Research Gap Finder",
        "ImageGen":  "🎨 Image Generation",
    }
    am = st.session_state["active_mode"]
    st.caption("Current mode")
    st.markdown(f"**{mode_labels.get(am, am)}**")

    if am != "auto":
        if st.button("↩ Back to Auto-Route"):
            st.session_state["active_mode"] = "auto"
            st.rerun()

    st.divider()

    # Knowledge base status
    if st.session_state["vector_store"]:
        st.success("📚 Knowledge base loaded")
        if st.button("🗑️ Clear KB"):
            st.session_state["vector_store"] = None
            st.session_state["pending_file"] = None
            st.rerun()
    else:
        st.info("No document loaded")

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.session_state["pending_file"] = None
        clear_agent_history("Wolf")
        st.rerun()

    msgs = st.session_state["messages"]
    if msgs:
        export = "\n\n---\n\n".join(
            f"**{'You' if m['role']=='user' else 'Wolf Scholar'}:** {m['content']}"
            for m in msgs
        )
        st.download_button("💾 Export chat", data=export,
                           file_name="wolf_scholar_chat.md", mime="text/markdown")

    st.divider()
    st.caption("Wolf Scholar v2.0 · Groq Llama 3.3\nArXiv · FAISS · DuckDuckGo")


# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="ws-header">
  <div class="ws-title">🐺 Wolf Scholar</div>
  <div class="ws-sub">Multi-Modal AI Research Copilot &nbsp;·&nbsp; Attach a PDF or Image to activate specialized reasoning</div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
#  HELPERS
# ============================================================
def render_chunks(chunks):
    for i, c in enumerate(chunks, 1):
        rel = c.get("relevance", 0)
        src = c.get("source", "?")
        pg  = c.get("page")
        pg_s = f" · p.{pg+1}" if pg is not None else ""
        st.markdown(f"**Chunk {i}** — `{rel}%` match · *{src}*{pg_s}")
        st.code(c["content"], language=None)
        if i < len(chunks): st.divider()


def tool_badge(text: str):
    st.markdown(f'<div class="ws-tool-badge">⚙ {text}</div>', unsafe_allow_html=True)


def file_is_image(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

def file_is_doc(name: str) -> bool:
    return name.lower().endswith((".pdf", ".txt"))


# ============================================================
#  CHAT HISTORY
# ============================================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool"):
            tool_badge(msg["tool"])
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"])
        if msg.get("chunks"):
            with st.expander("📂 Retrieved source fragments"):
                render_chunks(msg["chunks"])


# ============================================================
#  INPUT AREA — file uploader + ➕ popover + chat input
# ============================================================

# --- Pending file badge ---
pf = st.session_state["pending_file"]
if pf:
    icon = "🖼️" if file_is_image(pf["name"]) else "📄"
    route = "Vision Analysis" if file_is_image(pf["name"]) else "Knowledge Base (RAG)"
    st.markdown(
        f'<div class="ws-attach-preview">{icon} <strong>{pf["name"]}</strong>'
        f'&nbsp;→&nbsp; will route to <em>{route}</em>'
        f'&nbsp;&nbsp;<span style="cursor:pointer;color:#ef4444;">✕ clear</span></div>',
        unsafe_allow_html=True,
    )

# --- Toolbar: [➕] + file uploader ---
left_col, right_col = st.columns([0.07, 0.93])

with left_col:
    with st.popover("➕", help="Advanced Research Tools"):
        st.markdown("##### 🔧 Advanced Research Tools")
        st.caption("Override auto-routing for specialized tasks")
        for m_key, m_emoji, m_label, m_desc in ADVANCED_MODES:
            is_active = (st.session_state["active_mode"] == m_key)
            btn_label = f"{m_emoji} {m_label}" + (" ✓" if is_active else "")
            if st.button(btn_label, key=f"adv_{m_key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["active_mode"] = m_key
                st.rerun()
            st.caption(f"  {m_desc}")

        if st.session_state["active_mode"] != "auto":
            if st.button("↩ Auto-Route", key="back_auto", use_container_width=True):
                st.session_state["active_mode"] = "auto"
                st.rerun()

# ============================================================
#  CHAT INPUT + ROUTING ENGINE
# ============================================================
with right_col:
    # --- Combined File Uploader directly above the input ---
    st.markdown("##### ➕ Attach PDF or Image")
    up = st.file_uploader("Attach a file", type=["pdf", "txt", "png", "jpg", "jpeg"],
                          key="wolf_scholar_attach", label_visibility="collapsed")
    if up:
        st.session_state["pending_file"] = {
            "name":  up.name,
            "bytes": up.read(),
            "type":  up.type,
        }
        if file_is_doc(up.name):
            # Pre-process PDFs immediately
            with st.spinner(f"Loading {up.name} into knowledge base..."):
                from agents.rag_agent import build_vector_store
                st.session_state["vector_store"] = build_vector_store([up])
            st.success(f"✅ {up.name} loaded!")
        elif file_is_image(up.name):
            st.image(up, caption=up.name, use_container_width=True)
        st.rerun()

    if st.session_state["pending_file"]:
        if st.button("🗑️ Remove file", key="rm_pending"):
            st.session_state["pending_file"] = None
            st.rerun()

    mode   = st.session_state["active_mode"]
    hints  = {
        "auto":      "Ask anything — attach a PDF or image above for specialized analysis...",
        "ArXiv":     "Search academic papers... e.g. 'attention mechanisms efficiency'",
        "LitReview": "Topic for literature review... e.g. 'LLM hallucination detection'",
        "GapFinder": "Field to analyze... e.g. 'federated learning in healthcare'",
        "ImageGen":  "Describe your image... e.g. 'A wolf in a quantum research lab'",
    }
    prompt = st.chat_input(hints.get(mode, "Ask Wolf anything..."))

if prompt:
    pf   = st.session_state["pending_file"]
    mode = st.session_state["active_mode"]

    # Display user message
    with st.chat_message("user"):
        if pf:
            st.markdown(f"📎 *{pf['name']}*\n\n{prompt}")
        else:
            st.markdown(prompt)

    user_content = f"📎 *{pf['name']}*\n\n{prompt}" if pf else prompt
    st.session_state["messages"].append({"role": "user", "content": user_content})
    save_message("Wolf", "user", user_content)

    # ── ROUTING ENGINE ──
    with st.chat_message("assistant"):
        response    = ""
        tool_used   = ""
        extra_chunks = None
        extra_image  = None

        try:

            # ============================================================
            #  ADVANCED MODE OVERRIDES
            # ============================================================
            if mode == "ArXiv":
                tool_used = "ArXiv Scholar · Searching 2M+ academic papers"
                with st.status("🔬 Searching ArXiv database...", expanded=True) as s:
                    papers = search_arxiv(prompt, max_results=5)
                    s.update(label=f"✅ Found {len(papers)} papers", state="complete", expanded=False)

                if papers:
                    st.markdown("**📄 Papers Retrieved:**")
                    for i, p in enumerate(papers, 1):
                        with st.expander(f"[{i}] {p['title']} ({p['year']})", expanded=False):
                            st.caption(f"{p['authors']} · {p['categories']}")
                            st.write(p["abstract"])
                            c1, c2 = st.columns(2)
                            c1.markdown(f"[🔗 ArXiv]({p['url']})")
                            c2.markdown(f"[📥 PDF]({p['pdf_url']})")
                    st.markdown("**🧠 Synthesis:**")
                    llm = get_llm()
                    response = st.write_stream(
                        c.content for c in llm.stream([
                            SystemMessage(content=(
                                "Synthesize these ArXiv papers clearly."
                                " Cite each as [[Title]](url). End with Key Takeaways."
                            )),
                            HumanMessage(content=f"Query: {prompt}\n\nPapers:\n{format_papers_for_llm(papers)}"),
                        ]) if c.content
                    )
                else:
                    response = "⚠️ No matching papers found. Try broader search terms."
                    st.markdown(response)

            elif mode == "LitReview":
                tool_used = "Literature Review · ArXiv + DuckDuckGo synthesis"
                with st.status("📖 Gathering sources for literature review...", expanded=True) as s:
                    papers = search_arxiv(prompt, max_results=6)
                    s.update(label=f"✅ {len(papers)} papers indexed", state="complete", expanded=False)
                response = st.write_stream(literature_review_stream(prompt))

            elif mode == "GapFinder":
                tool_used = "Research Gap Finder · Frontier analysis"
                with st.status("🕵️ Mapping the research frontier...", expanded=True) as s:
                    papers = search_arxiv(prompt, max_results=8)
                    s.update(label=f"✅ Analyzed {len(papers)} frontier papers", state="complete", expanded=False)
                response = st.write_stream(research_gap_stream(prompt))

            elif mode == "ImageGen":
                tool_used = "Image Generation"
                with st.status("🎨 Generating image...", expanded=True) as s:
                    gen_img = generate_image(prompt)
                    s.update(label="✅ Image generated", state="complete", expanded=False)
                st.image(gen_img)
                response    = f"🎨 Generated: *{prompt}*"
                extra_image = gen_img
                st.markdown(response)

            # ============================================================
            #  AUTO-ROUTE by attached file type
            # ============================================================
            elif pf and file_is_doc(pf["name"]):
                # ── PDF / TXT → RAG ──
                tool_used = f"Knowledge Base · FAISS semantic search on {pf['name']}"
                if st.session_state["vector_store"] is None:
                    response = "⚠️ Document not yet loaded into memory. Please re-attach it."
                    st.markdown(response)
                else:
                    with st.status(f"📚 Wolf Scholar is reading the PDF via FAISS...", expanded=True) as s:
                        st.write(f"**Document:** `{pf['name']}`")
                        st.write(f"**Query:** `{prompt}`")
                        chunks = get_retrieved_chunks(prompt, st.session_state["vector_store"])
                        s.update(label=f"✅ Retrieved {len(chunks)} relevant chunks", state="complete", expanded=False)
                    tool_badge(tool_used)
                    response      = st.write_stream(
                        ask_stream(prompt, st.session_state["vector_store"], st.session_state["messages"][:-1])
                    )
                    extra_chunks = chunks
                    with st.expander("📂 Source fragments"):
                        render_chunks(chunks)

            elif pf and file_is_image(pf["name"]):
                # ── Image → Vision OCR ──
                tool_used = f"Vision · Analyzing image pixels of {pf['name']}"
                with st.status("👁️ Wolf Scholar is analyzing the image pixels...", expanded=True) as s:
                    st.write(f"**Image:** `{pf['name']}`")
                    extracted = perform_ocr(pf["bytes"])
                    s.update(label="✅ Image content extracted", state="complete", expanded=False)
                tool_badge(tool_used)
                llm = get_llm()
                response = st.write_stream(
                    c.content for c in llm.stream([
                        SystemMessage(content=(
                            "You are an expert image and document analyst. "
                            "Use the extracted image content below to answer the user's question thoroughly.\n\n"
                            f"Extracted content:\n{extracted}"
                        )),
                        HumanMessage(content=prompt),
                    ]) if c.content
                )

            else:
                # ── No file → Web Search ──
                tool_used = "Web Search · DuckDuckGo"
                with st.status("🌐 Wolf Scholar is searching the web...", expanded=True) as s:
                    st.write(f"**Query:** `{prompt}`")
                    ddg    = DuckDuckGoSearchRun()
                    web_r  = ddg.run(prompt)
                    s.update(label="✅ Web search complete", state="complete", expanded=False)
                tool_badge(tool_used)
                llm = get_llm()
                response = st.write_stream(
                    c.content for c in llm.stream([
                        SystemMessage(content=(
                            "You are a precise, concise research assistant. "
                            "Summarize the search results below clearly. "
                            "Cite any URLs present in the results. Be helpful and direct.\n\n"
                            f"Search results:\n{web_r}"
                        )),
                        HumanMessage(content=prompt),
                    ]) if c.content
                )

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                response = "⏳ Rate limit reached — please wait a moment and retry."
            elif "401" in err or "api_key" in err.lower():
                response = "🔑 API key error — check your `.env` file."
            else:
                response = f"Something went wrong: {err}"
            st.error(response)

    # Save to session + DB
    msg_record = {
        "role":    "assistant",
        "content": response or "",
        "tool":    tool_used,
    }
    if extra_chunks: msg_record["chunks"] = extra_chunks
    if extra_image:  msg_record["image"]  = extra_image
    st.session_state["messages"].append(msg_record)
    save_message("Wolf", "assistant", response or "", {"tool": tool_used})

    # Clear pending file after use
    st.session_state["pending_file"] = None

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div style='text-align:center;color:#374151;font-size:0.72rem;
            border-top:1px solid #23252a;padding:1rem 0 0;margin-top:2rem;'>
  🐺 Wolf Scholar v2.0 &nbsp;·&nbsp; Groq Llama 3.3 70B &nbsp;·&nbsp; ArXiv · FAISS · DuckDuckGo<br>
  Deployed by <strong style='color:#4CAF50;'>Abdullah Ibne Tayeb Tamur</strong>
</div>
""", unsafe_allow_html=True)
