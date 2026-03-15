"""
app/app_ui.py — Wolf Scholar: Intelligence Terminal v2
Entry:  streamlit run app/app_ui.py
"""
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

from agents.router import route
from database.vector_store import init_db, load_history, save_message

load_dotenv()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Wolf Scholar",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
  --bg:      #080B12;
  --bg2:     #0D1120;
  --surface: #141B2D;
  --surf2:   #1A2340;
  --cyan:    #00E5FF;
  --violet:  #9D6FFF;
  --amber:   #FFB830;
  --green:   #00FF87;
  --red:     #FF4D6D;
  --text:    #E2E8F0;
  --muted:   #64748B;
  --border:  rgba(0,229,255,0.1);
}

/* ── Chrome reset ── */
html, body, [class*="css"], .stApp {
  font-family: 'JetBrains Mono', monospace !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
#MainMenu, header, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] {
  visibility: hidden !important; display: none !important;
}
[data-testid="stHeader"] { height: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 4px; }

/* ── Animations ── */
@keyframes topflow {
  0%  { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100%{ background-position: 0% 50%; }
}
@keyframes pulse      { 0%,100%{opacity:.3} 50%{opacity:.8} }
@keyframes float      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes fadeSlideUp{ from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes geoSpin    { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
@keyframes typeBounce { 0%,80%,100%{transform:scale(0);opacity:.4} 40%{transform:scale(1);opacity:1} }
@keyframes scanline   { 0%{top:-2%} 100%{top:102%} }
@keyframes glowPulse  { 0%,100%{box-shadow:0 0 12px rgba(0,229,255,.4)} 50%{box-shadow:0 0 28px rgba(0,229,255,.9)} }
@keyframes shimmer    { 0%{background-position:0% center} 100%{background-position:200% center} }

/* ── Fixed background layers ── */
.ws-dot-grid {
  position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image: radial-gradient(circle, rgba(0,229,255,.07) 1px, transparent 1px);
  background-size: 32px 32px;
}
.ws-geo { position:fixed; border-radius:50%; pointer-events:none; z-index:0; border:1px solid rgba(0,229,255,.04); }
.ws-geo-1 { width:500px;height:500px; top:-100px; right:-150px; animation:geoSpin 80s linear infinite; }
.ws-geo-2 { width:300px;height:300px; bottom:-50px; left:-80px;  animation:geoSpin 60s linear infinite reverse; }
.ws-geo-3 { width:200px;height:200px; top:40%;     left:10%;     animation:geoSpin 70s linear infinite; }
.ws-scan  {
  position:fixed; left:0; right:0; height:2px; z-index:1; pointer-events:none;
  background:linear-gradient(transparent, rgba(0,229,255,.05), transparent);
  animation:scanline 8s linear infinite;
}
.ws-top-bar {
  position:fixed; top:0; left:0; right:0; height:2px; z-index:9999;
  background:linear-gradient(90deg, transparent, var(--cyan), var(--violet), var(--amber), var(--cyan), transparent);
  background-size:300%; animation:topflow 4s ease infinite;
}

/* ── Layout ── */
.block-container {
  max-width: 860px !important;
  padding: 0 1.2rem 10rem 1.2rem !important;
  margin: 0 auto !important;
  position: relative; z-index: 2;
}

/* ── Header ── */
.ws-header {
  display:flex; align-items:center; justify-content:space-between;
  padding: .9rem 1.4rem;
  background: rgba(8,11,18,.92);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.4rem;
  position: sticky; top: 0; z-index: 50;
  animation: fadeSlideUp .5s ease;
}
.ws-logo { display:flex; align-items:center; gap:12px; }
.ws-icon {
  width:38px; height:38px; border-radius:10px;
  background:linear-gradient(135deg,var(--cyan),var(--violet));
  display:flex; align-items:center; justify-content:center;
  font-size:20px; animation:glowPulse 3s ease-in-out infinite;
}
.ws-title {
  font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
  background:linear-gradient(90deg,var(--cyan),var(--violet));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  letter-spacing:-.5px;
}
.ws-sub { font-size:.65rem; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase; }
.ws-pill {
  display:flex; align-items:center; gap:6px;
  padding:4px 12px; border-radius:20px;
  background:rgba(0,255,135,.08); border:1px solid rgba(0,255,135,.2);
  font-size:.7rem; color:var(--green);
}
.ws-dot { width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite; }
.ws-credit { font-size:.62rem; color:var(--muted); text-align:right; line-height:1.4; }
.ws-credit span { color:rgba(0,229,255,.5); }

/* ── Welcome ── */
.ws-welcome { text-align:center; padding:3rem 0 2rem; animation:fadeSlideUp .6s ease; }
.ws-big-wolf { font-size:4.5rem; display:block; margin-bottom:.8rem; animation:float 4s ease-in-out infinite; }
.ws-big-title {
  font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
  background:linear-gradient(90deg,var(--cyan) 0%,var(--violet) 50%,var(--cyan) 100%);
  background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation:shimmer 4s linear infinite; letter-spacing:-1px;
}
.ws-big-sub { color:var(--muted); font-size:.82rem; letter-spacing:2px; text-transform:uppercase; margin:8px 0 2rem; }

/* ── Capability cards ── */
.ws-card {
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:1.2rem; cursor:pointer; transition:all .2s ease; animation:fadeSlideUp .5s ease;
}
.ws-card:hover {
  border-color:rgba(0,229,255,.3); background:var(--surf2);
  transform:translateY(-3px); box-shadow:0 8px 30px rgba(0,229,255,.08);
}
.ws-card-icon { font-size:1.8rem; margin-bottom:.5rem; }
.ws-card-title { font-family:'Syne',sans-serif; font-size:.92rem; font-weight:700; color:var(--text); margin-bottom:3px; }
.ws-card-desc  { font-size:.72rem; color:var(--muted); }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
  border-radius:16px !important; padding:.9rem 1.1rem !important;
  margin-bottom:.6rem !important; animation:fadeSlideUp .35s ease;
}
[data-testid="stChatMessageUser"] {
  background:var(--surf2) !important;
  border:1px solid rgba(0,229,255,.15) !important;
  border-radius:16px 4px 16px 16px !important;
  margin-left:10% !important;
}
[data-testid="stChatMessageAssistant"] {
  background:var(--surface) !important;
  border:1px solid var(--border) !important;
  border-radius:4px 16px 16px 16px !important;
  box-shadow:0 0 30px rgba(0,229,255,.05) !important;
  margin-right:10% !important;
}

/* ── Agent badges ── */
.badge {
  display:inline-block; padding:2px 12px; border-radius:999px;
  font-size:.68rem; letter-spacing:.5px; margin-top:8px;
  font-family:'JetBrains Mono',monospace;
}
.badge-rag    { background:rgba(0,229,255,.1);  color:var(--cyan);   border:1px solid rgba(0,229,255,.3);  }
.badge-vision { background:rgba(157,111,255,.1); color:var(--violet); border:1px solid rgba(157,111,255,.3);}
.badge-search { background:rgba(255,184,48,.1);  color:var(--amber);  border:1px solid rgba(255,184,48,.3); }
.badge-image  { background:rgba(0,255,135,.1);   color:var(--green);  border:1px solid rgba(0,255,135,.3);  }

/* ── Input ── */
[data-testid="stChatInput"] {
  backdrop-filter:blur(20px);
  background:rgba(8,11,18,.95) !important;
  border-top:1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea {
  background:var(--surface) !important; border:1px solid #1A2340 !important;
  border-radius:12px !important; color:var(--text) !important;
  font-family:'JetBrains Mono',monospace !important; font-size:.86rem !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color:rgba(0,229,255,.3) !important;
  box-shadow:0 0 0 3px rgba(0,229,255,.06) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
  background:var(--surface) !important; border:1px dashed #1A2340 !important; border-radius:10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span { color:var(--muted) !important; font-size:.75rem !important; }

/* ── Status ── */
[data-testid="stStatus"] {
  background:var(--bg2) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; font-size:.8rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background:rgba(8,11,18,.85) !important;
  backdrop-filter:blur(12px) !important;
  border-right:1px solid var(--border) !important;
}
.sb-agent {
  display:flex; align-items:center; gap:10px; padding:8px 10px;
  border-radius:8px; margin-bottom:6px; font-size:.78rem; color:var(--muted);
  border:1px solid transparent; transition:all .15s;
}
.sb-agent:hover { background:var(--surface); border-color:var(--border); color:var(--text); }
.sb-icon { width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:13px; }

/* ── Misc ── */
hr { border-color:var(--border) !important; margin:.6rem 0 !important; }
.stButton > button {
  background:var(--surface) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
  font-family:'JetBrains Mono',monospace !important; font-size:.78rem !important;
  transition:all .15s ease !important;
}
.stButton > button:hover { border-color:rgba(0,229,255,.4) !important; box-shadow:0 0 12px rgba(0,229,255,.1) !important; }
.attach-label { font-size:.65rem; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px; }
</style>

<!-- Background layers injected once -->
<div class="ws-dot-grid"></div>
<div class="ws-geo ws-geo-1"></div>
<div class="ws-geo ws-geo-2"></div>
<div class="ws-geo ws-geo-3"></div>
<div class="ws-scan"></div>
<div class="ws-top-bar"></div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE  (initialised before any widget renders)
# ════════════════════════════════════════════════════════════════════════════
init_db()

defaults = {
    "messages":     [],
    "vector_store": None,
    "upload_key":   0,      # bumped by New Chat to force file-uploader reset
    "history_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load SQLite history once per session
if not st.session_state.history_loaded:
    try:
        st.session_state.messages = load_history()
    except Exception:
        pass
    st.session_state.history_loaded = True


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .5rem;text-align:center;'>
      <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                  background:linear-gradient(90deg,#00E5FF,#9D6FFF);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🐺 Wolf Scholar
      </div>
      <div style='font-size:.62rem;color:#64748B;letter-spacing:1px;margin-top:2px;'>Intelligence Terminal</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:.65rem;color:#64748B;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Agents Online</div>",
                unsafe_allow_html=True)

    for icon, lbl, color, sub in [
        ("📄", "RAG · Literature",  "#00E5FF", "PDF → FAISS"),
        ("🔬", "Vision · Pathology","#9D6FFF",  "Image → Llama-4"),
        ("🔍", "Search · Academic", "#FFB830",  "arXiv · IEEE"),
        ("🎨", "Image Generation",  "#00FF87",  "Pollinations flux"),
    ]:
        st.markdown(f"""
        <div class="sb-agent">
          <div class="sb-icon" style="background:rgba(0,0,0,.3);border:1px solid {color}33;">{icon}</div>
          <div>
            <div style="color:#E2E8F0;font-size:.76rem;">{lbl}</div>
            <div style="font-size:.62rem;color:#64748B;">{sub}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    if st.session_state.vector_store:
        st.success("📚 Knowledge base active")
    else:
        st.caption("No document loaded")

    st.divider()

    # ── New Chat ──────────────────────────────────────────────────────────────
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.vector_store = None
        st.session_state.history_loaded = True   # don't reload from SQLite
        # Increment key → forces Streamlit to destroy and recreate file uploader
        st.session_state.upload_key  += 1
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:.62rem;color:#3A3A5A;text-align:center;padding-top:4px;'>
      Engineered &amp; Deployed by<br>
      <span style='color:rgba(0,229,255,.4);'>Abdullah Ibne Tayeb Tamur</span>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="ws-header">
  <div class="ws-logo">
    <div class="ws-icon">🐺</div>
    <div>
      <div class="ws-title">Wolf Scholar</div>
      <div class="ws-sub">Academic Intelligence · Diagnostic Vision · Live Research</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <div class="ws-pill"><div class="ws-dot"></div>Systems Online</div>
    <div class="ws-credit">
      Engineered &amp; Deployed by<br>
      <span>Abdullah Ibne Tayeb Tamur</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN  (shown only when chat is empty)
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <div class="ws-welcome">
      <span class="ws-big-wolf">🐺</span>
      <div class="ws-big-title">Wolf Scholar</div>
      <div class="ws-big-sub">Precision Intelligence for Academic &amp; Diagnostic Research</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for icon, title, desc, col in [
        ("📄", "Literature Review",  "Upload PDF · FAISS retrieval · Research gaps",            col1),
        ("🔬", "Diagnostic Vision",  "Upload image · Pathology · Precision agriculture",         col2),
        ("🔍", "Live Research",      "arXiv · Nature · IEEE · Real-time academic synthesis",     col1),
        ("🎨", "Image Generation",   'Say "generate…" · Pollinations AI · flux model',           col2),
    ]:
        with col:
            st.markdown(f"""
            <div class="ws-card">
              <div class="ws-card-icon">{icon}</div>
              <div class="ws-card-title">{title}</div>
              <div class="ws-card-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_image"):
            # Stored as bytes in session; re-display
            st.image(msg["content"])
            if msg.get("caption"):
                st.caption(msg["caption"])
        else:
            st.markdown(msg.get("content", ""))
        if msg.get("badge"):
            st.markdown(
                f'<span class="badge badge-{msg["badge_type"]}">{msg["badge"]}</span>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
#  INPUT BAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="attach-label">➕ Attach PDF or Image — system auto-routes</div>',
            unsafe_allow_html=True)

# Dynamic key resets the widget when New Chat is clicked
_upload_key = f"ws_up_{st.session_state.upload_key}"
uploaded_file = st.file_uploader(
    "Attach",
    type=["pdf", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key=_upload_key,
)

# Preview — read bytes, then seek back so router can read them too
_file_bytes: bytes | None = None
if uploaded_file is not None:
    _file_bytes = uploaded_file.read()   # read once
    uploaded_file.seek(0)               # rewind for router

    if uploaded_file.type.startswith("image/"):
        st.image(_file_bytes, width=240, caption=f"🔬 Ready · {uploaded_file.name}")
    else:
        st.markdown(
            f'<div style="font-size:.78rem;color:var(--cyan);padding:6px 0;">📄 {uploaded_file.name} ready for RAG</div>',
            unsafe_allow_html=True,
        )

user_input = st.chat_input("Ask Wolf Scholar anything...")


# ════════════════════════════════════════════════════════════════════════════
#  STATUS LABELS
# ════════════════════════════════════════════════════════════════════════════
_STATUS_RUNNING = {
    "rag":    "🐺 Wolf Scholar is parsing the research paper...",
    "vision": "🔬 Wolf Scholar is analyzing the pathology image...",
    "search": "🔍 Wolf Scholar is searching arXiv, Nature & IEEE...",
    "image":  "🎨 Wolf Scholar is generating the visualization...",
}
_STATUS_DONE = {
    "rag":    "✅ Research parsed",
    "vision": "✅ Image analysis complete",
    "search": "✅ Academic search complete",
    "image":  "✅ Image generated",
}


# ════════════════════════════════════════════════════════════════════════════
#  ROUTING & RESPONSE HANDLER
# ════════════════════════════════════════════════════════════════════════════
if user_input:

    # ── 1. User bubble ────────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── 2. Route + respond ────────────────────────────────────────────────────
    vs_ref        = [st.session_state.vector_store]
    full_response = ""
    is_img        = False
    img_bytes_out: bytes | None = None
    img_prompt    = ""
    rtype         = "search"   # safe default
    result: dict  = {}         # always defined — prevents NameError on early errors

    with st.chat_message("assistant"):

        # Everything — routing, processing, display — lives inside ONE status
        # block so .update() works and no orphaned widget is created.
        with st.status("🐺 Wolf Scholar is working...", expanded=True) as status:
            try:
                result = route(user_input, uploaded_file, vs_ref)
                rtype  = result["type"]
                status.update(label=_STATUS_RUNNING[rtype], state="running")

                # ──────── process each route ──────────────────────────────────
                if rtype == "image":
                    status.write("🎨 Calling Pollinations AI flux model...")
                    img_bytes_out = result["content"]
                    img_prompt    = result.get("prompt", user_input)
                    full_response = f"[Image] Prompt: {img_prompt}"
                    is_img = True

                elif rtype == "vision":
                    status.write("🔬 Running Llama-4 Scout vision model...")
                    full_response = result["content"]   # already a string

                else:   # rag or search — generator
                    status.write("⚙️ Collecting streamed response...")
                    chunks = list(result["content"])    # consume generator
                    full_response = "".join(chunks)

                status.update(
                    label=_STATUS_DONE[rtype],
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate_limit" in err.lower():
                    full_response = "⏳ Rate limit hit — wait a moment and retry."
                elif "401" in err or "api_key" in err.lower():
                    full_response = "🔑 API key error — check your `.env` file."
                else:
                    full_response = f"⚠️ {err}"
                status.update(label="❌ Error", state="error", expanded=True)
                status.write(full_response)

        # ── Display result (outside status, still inside assistant bubble) ────
        if is_img and img_bytes_out:
            st.image(img_bytes_out, caption="Pollinations AI · flux model")
            st.caption(f"Prompt: {img_prompt}")
        elif full_response:
            st.markdown(full_response)

        # Agent badge — result always defined (even if {})
        badge_label = result.get("label", "")
        if badge_label:
            st.markdown(
                f'<span class="badge badge-{rtype}">{badge_label}</span>',
                unsafe_allow_html=True,
            )

    # ── Cache updated vector store ────────────────────────────────────────────
    if vs_ref[0] is not None:
        st.session_state.vector_store = vs_ref[0]

    # ── Persist to session state + SQLite ─────────────────────────────────────
    record: dict = {
        "role":       "assistant",
        "content":    full_response,
        "badge":      result.get("label", ""),
        "badge_type": rtype,
    }
    if is_img and img_bytes_out:
        record["is_image"] = True
        record["content"]  = img_bytes_out
        record["caption"]  = f"Prompt: {img_prompt}"

    st.session_state.messages.append(record)
    save_message("assistant",
                 full_response if not is_img else "[Generated Image]")


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;color:#1E2A3A;font-size:.65rem;
            border-top:1px solid rgba(0,229,255,.05);
            padding:1.2rem 0 7rem;margin-top:2rem;
            font-family:JetBrains Mono,monospace;'>
  🐺 Wolf Scholar v2 &nbsp;·&nbsp; Groq Llama 3.3 70B &nbsp;·&nbsp;
  Llama-4 Scout &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Pollinations flux &nbsp;·&nbsp; LangSmith<br>
  <span style='color:rgba(0,229,255,.2);'>
    Precision Intelligence for Academic &amp; Diagnostic Research
    &nbsp;·&nbsp; Engineered by Abdullah Ibne Tayeb Tamur
  </span>
</div>
""", unsafe_allow_html=True)
