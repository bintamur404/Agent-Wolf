"""
app/app_ui.py ─ Wolf Scholar: Intelligence Terminal v2
Entry point: streamlit run app/app_ui.py
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
from agents.router import route
from database.vector_store import init_db, save_message, load_history

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
#  PREMIUM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── CSS Variables ── */
:root {
  --bg:      #080B12;
  --bg2:     #0D1120;
  --surface: #141B2D;
  --surface2:#1A2340;
  --cyan:    #00E5FF;
  --violet:  #9D6FFF;
  --amber:   #FFB830;
  --green:   #00FF87;
  --red:     #FF4D6D;
  --text:    #E2E8F0;
  --muted:   #64748B;
  --border:  rgba(0,229,255,0.1);
}

/* ── Reset & chrome removal ── */
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
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 4px; }

/* ── All 8 Keyframe Animations ── */
@keyframes topflow {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50%       { opacity: 0.8; }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-8px); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes geoSpin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes typeBounce {
  0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
  40%           { transform: scale(1); opacity: 1; }
}
@keyframes scanline {
  0%   { top: -2%; }
  100% { top: 102%; }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 12px rgba(0,229,255,0.4); }
  50%       { box-shadow: 0 0 28px rgba(0,229,255,0.9); }
}
@keyframes shimmer {
  0%   { background-position: 0% center; }
  100% { background-position: 200% center; }
}
@keyframes spin-slow { to { transform: rotate(360deg); } }

/* ── Fixed background elements ── */
.ws-bg-layers {
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
}
.ws-dot-grid {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle, rgba(0,229,255,0.07) 1px, transparent 1px);
  background-size: 32px 32px;
}
.ws-geo-1, .ws-geo-2, .ws-geo-3 {
  position: fixed; border-radius: 50%; pointer-events: none; z-index: 0;
  border: 1px solid rgba(0,229,255,0.04);
}
.ws-geo-1 { width:500px; height:500px; top:-100px; right:-150px; animation: geoSpin 80s linear infinite; }
.ws-geo-2 { width:300px; height:300px; bottom:-50px; left:-80px;  animation: geoSpin 60s linear infinite reverse; }
.ws-geo-3 { width:200px; height:200px; top:40%; left:10%;         animation: geoSpin 70s linear infinite; }
.ws-scanline {
  position: fixed; left: 0; right: 0; height: 2px; z-index: 1; pointer-events: none;
  background: linear-gradient(transparent, rgba(0,229,255,0.05), transparent);
  animation: scanline 8s linear infinite;
}
.ws-top-border {
  position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 999;
  background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), var(--amber), var(--cyan), transparent);
  background-size: 300%; animation: topflow 4s ease infinite;
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
  position: sticky; top: 0; z-index: 50;
  backdrop-filter: blur(20px);
  background: rgba(8,11,18,0.92);
  border-bottom: 1px solid var(--border);
  padding: 0.9rem 1.4rem;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1.5rem;
  animation: fadeSlideUp 0.5s ease;
}
.ws-logo-group { display: flex; align-items: center; gap: 12px; }
.ws-wolf-icon {
  width: 38px; height: 38px; border-radius: 10px;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; animation: glowPulse 3s ease-in-out infinite;
}
.ws-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem; font-weight: 800;
  background: linear-gradient(90deg, var(--cyan), var(--violet));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.ws-subtitle {
  font-size: 0.65rem; color: var(--muted);
  letter-spacing: 1.5px; text-transform: uppercase;
}
.ws-status-pill {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 20px;
  background: rgba(0,255,135,0.08);
  border: 1px solid rgba(0,255,135,0.2);
  font-size: 0.7rem; color: var(--green);
}
.ws-status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); animation: pulse 2s ease-in-out infinite;
}
.ws-credit {
  font-size: 0.62rem; color: var(--muted);
  font-family: 'JetBrains Mono', monospace; text-align: right;
  line-height: 1.4;
}
.ws-credit span { color: rgba(0,229,255,0.5); }

/* ── Welcome screen ── */
.ws-welcome {
  text-align: center; padding: 3rem 0 2rem;
  animation: fadeSlideUp 0.6s ease;
}
.ws-welcome-wolf {
  font-size: 4.5rem; display: block; margin-bottom: 0.8rem;
  animation: float 4s ease-in-out infinite;
}
.ws-welcome-title {
  font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
  background: linear-gradient(90deg, var(--cyan) 0%, var(--violet) 50%, var(--cyan) 100%);
  background-size: 200% auto; -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; animation: shimmer 4s linear infinite;
  letter-spacing: -1px;
}
.ws-welcome-sub {
  color: var(--muted); font-size: 0.82rem;
  letter-spacing: 2px; text-transform: uppercase; margin: 8px 0 2rem;
}

/* ── Capability cards ── */
.ws-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.2rem;
  transition: all 0.2s ease; cursor: pointer;
  animation: fadeSlideUp 0.5s ease;
}
.ws-card:hover {
  border-color: rgba(0,229,255,0.3);
  background: var(--surface2);
  transform: translateY(-3px);
  box-shadow: 0 8px 30px rgba(0,229,255,0.08);
}
.ws-card-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.ws-card-title {
  font-family: 'Syne', sans-serif; font-size: 0.92rem;
  font-weight: 700; color: var(--text); margin-bottom: 3px;
}
.ws-card-desc { font-size: 0.72rem; color: var(--muted); }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  border-radius: 16px !important;
  padding: 0.9rem 1.1rem !important;
  margin-bottom: 0.6rem !important;
  animation: fadeSlideUp 0.35s ease;
}
[data-testid="stChatMessageUser"] {
  background: var(--surface2) !important;
  border: 1px solid rgba(0,229,255,0.15) !important;
  border-radius: 16px 4px 16px 16px !important;
  margin-left: 10% !important;
}
[data-testid="stChatMessageAssistant"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px 16px 16px 16px !important;
  box-shadow: 0 0 30px rgba(0,229,255,0.05) !important;
  margin-right: 10% !important;
}

/* ── Agent badge pills ── */
.agent-badge {
  display: inline-block; padding: 2px 12px;
  border-radius: 999px; font-size: 0.68rem;
  letter-spacing: 0.5px; margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
}
.badge-rag    { background: rgba(0,229,255,0.1);  color: var(--cyan);   border: 1px solid rgba(0,229,255,0.3);  }
.badge-vision { background: rgba(157,111,255,0.1); color: var(--violet); border: 1px solid rgba(157,111,255,0.3);}
.badge-search { background: rgba(255,184,48,0.1);  color: var(--amber);  border: 1px solid rgba(255,184,48,0.3); }
.badge-image  { background: rgba(0,255,135,0.1);   color: var(--green);  border: 1px solid rgba(0,255,135,0.3);  }

/* ── Attach label ── */
.ws-attach-label {
  font-size: 0.65rem; color: var(--muted);
  letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px;
}

/* ── st.status ── */
[data-testid="stStatus"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-size: 0.8rem !important;
}

/* ── Chat input (fixed bottom) ── */
[data-testid="stChatInput"] {
  backdrop-filter: blur(20px);
  background: rgba(8,11,18,0.95) !important;
  border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  border: 1px solid #1A2340 !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.86rem !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(0,229,255,0.3) !important;
  box-shadow: 0 0 0 3px rgba(0,229,255,0.06) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface) !important;
  border: 1px dashed #1A2340 !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--muted) !important; font-size: 0.75rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: rgba(8,11,18,0.85) !important;
  backdrop-filter: blur(12px) !important;
  border-right: 1px solid var(--border) !important;
}
.sidebar-agent {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 10px; border-radius: 8px; margin-bottom: 6px;
  font-size: 0.78rem; color: var(--muted);
  border: 1px solid transparent; transition: all 0.15s;
}
.sidebar-agent:hover {
  background: var(--surface); border-color: var(--border); color: var(--text);
}
.sidebar-icon {
  width: 28px; height: 28px; border-radius: 7px;
  display: flex; align-items: center; justify-content: center; font-size: 13px;
}

/* ── Misc ── */
hr { border-color: var(--border) !important; margin: 0.6rem 0 !important; }
.stButton > button {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: rgba(0,229,255,0.4) !important;
  box-shadow: 0 0 12px rgba(0,229,255,0.1) !important;
}
</style>

<!-- Fixed background layers (dot grid + geo circles + scanline + top border) -->
<div class="ws-dot-grid"></div>
<div class="ws-geo-1"></div>
<div class="ws-geo-2"></div>
<div class="ws-geo-3"></div>
<div class="ws-scanline"></div>
<div class="ws-top-border"></div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE & DB
# ════════════════════════════════════════════════════════════════════════════
init_db()

if "messages"     not in st.session_state: st.session_state.messages     = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

if not st.session_state.messages:
    try:
        st.session_state.messages = load_history()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem; text-align:center;'>
      <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                  background:linear-gradient(90deg,#00E5FF,#9D6FFF);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🐺 Wolf Scholar
      </div>
      <div style='font-size:0.62rem;color:#64748B;letter-spacing:1px;margin-top:2px;'>
        Intelligence Terminal
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:0.65rem;color:#64748B;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Agents</div>", unsafe_allow_html=True)
    for icon, label, color, desc in [
        ("📄", "RAG · Literature",  "#00E5FF", "PDF → FAISS"),
        ("🔬", "Vision · Pathology","#9D6FFF",  "Image → Llama-4"),
        ("🔍", "Search · Academic", "#FFB830",  "arXiv · IEEE"),
        ("🎨", "Image Generation",  "#00FF87",  "Pollinations flux"),
    ]:
        st.markdown(f"""
        <div class="sidebar-agent">
          <div class="sidebar-icon" style="background:rgba(0,0,0,0.3);border:1px solid {color}33;">{icon}</div>
          <div>
            <div style="color:#E2E8F0;font-size:0.76rem;">{label}</div>
            <div style="font-size:0.62rem;color:#64748B;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.session_state.vector_store:
        st.success("📚 Knowledge base active")
    else:
        st.caption("No document loaded")

    st.divider()
    if st.button("🗑 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        # Bump the key to force the file uploader to reset (Streamlit won't
        # let you set a file_uploader value directly via session_state)
        st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.62rem;color:#3A3A5A;text-align:center;padding-top:4px;'>
      Engineered & Deployed by<br>
      <span style='color:rgba(0,229,255,0.4);'>Abdullah Ibne Tayeb Tamur</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="ws-header">
  <div class="ws-logo-group">
    <div class="ws-wolf-icon">🐺</div>
    <div>
      <div class="ws-title">Wolf Scholar</div>
      <div class="ws-subtitle">Academic Intelligence · Diagnostic Vision · Live Research</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <div class="ws-status-pill">
      <div class="ws-status-dot"></div>
      Systems Online
    </div>
    <div class="ws-credit">
      Engineered &amp; Deployed by<br>
      <span>Abdullah Ibne Tayeb Tamur</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <div class="ws-welcome">
      <span class="ws-welcome-wolf">🐺</span>
      <div class="ws-welcome-title">Wolf Scholar</div>
      <div class="ws-welcome-sub">Precision Intelligence for Academic &amp; Diagnostic Research</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    cards = [
        ("📄", "Literature Review",  "Upload PDF · Find research gaps · FAISS semantic analysis", c1),
        ("🔬", "Diagnostic Vision",  "Upload JPG/PNG · Pathology · Precision agriculture", c2),
        ("🔍", "Live Research",      "arXiv · Nature · IEEE · Real-time academic synthesis", c1),
        ("🎨", "Image Generation",   "Pollinations AI · flux model · Research visualizations", c2),
    ]
    for icon, title, desc, col in cards:
        with col:
            st.markdown(f"""
            <div class="ws-card">
              <div class="ws-card-icon">{icon}</div>
              <div class="ws-card-title">{title}</div>
              <div class="ws-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_image"):
            st.image(msg["content"])
            if msg.get("caption"):
                st.caption(msg["caption"])
        else:
            st.markdown(msg["content"])
        if msg.get("badge"):
            st.markdown(
                f'<span class="agent-badge badge-{msg["badge_type"]}">{msg["badge"]}</span>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
#  INPUT AREA
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ws-attach-label">➕ Attach PDF or Image — auto-routes to correct agent</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Attach",
    type=["pdf", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
    # Dynamic key forces widget reset when New Chat is clicked
    key=f"ws_upload_{st.session_state.get('upload_key', 0)}",
)

if uploaded_file:
    if uploaded_file.type.startswith("image/"):
        # Read into bytes for preview, then seek back so router can read it again
        img_bytes_preview = uploaded_file.read()
        uploaded_file.seek(0)   # ← Bug 2 fix: re-wind so router gets full bytes
        st.image(img_bytes_preview, width=240,
                 caption=f"🔬 Ready · {uploaded_file.name}")
    else:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#00E5FF;padding:6px 0;">📄 {uploaded_file.name} — ready for RAG analysis</div>',
            unsafe_allow_html=True,
        )

user_input = st.chat_input("Ask Wolf Scholar anything...")


# ════════════════════════════════════════════════════════════════════════════
#  ROUTING & RESPONSE
# ════════════════════════════════════════════════════════════════════════════
STATUS_LABELS = {
    "rag":    "🐺 Wolf Scholar is parsing the research paper...",
    "vision": "🔬 Wolf Scholar is analyzing the pathology image...",
    "search": "🔍 Wolf Scholar is searching arXiv, Nature & IEEE...",
    "image":  "🎨 Wolf Scholar is generating the visualization...",
}
STATUS_DONE = {
    "rag":    "✅ Research paper parsed — streaming response",
    "vision": "✅ Image analysis complete",
    "search": "✅ Academic search complete — streaming response",
    "image":  "✅ Image generated by Pollinations flux",
}

if user_input:
    # 1 — user bubble
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    vs_ref = [st.session_state.vector_store]
    full_response   = ""
    response_is_img = False
    result_data     = None
    rtype           = "search"  # safe default

    # 2+3 — single status block containing both routing AND response display
    # (Bug 3 fix: keeping routing + streaming inside one with-block so
    #  status_box.update() works correctly and no orphaned widget is created)
    with st.chat_message("assistant"):
        with st.status("🐺 Wolf Scholar is working...", expanded=True) as status_box:
            try:
                # Route
                result = route(user_input, uploaded_file, vs_ref)
                rtype  = result["type"]
                status_box.update(label=STATUS_LABELS[rtype], state="running")

                # Run / stream inside the status so progress is visible
                if rtype == "image":
                    status_box.write("🎨 Calling Pollinations flux model...")
                    result_data     = result["content"]
                    full_response   = f"[Image generated] Prompt: {result.get('prompt', '')}"
                    response_is_img = True

                elif rtype == "vision":
                    status_box.write("🔬 Running Llama-4 Scout vision analysis...")
                    full_response = result["content"]  # already a string (non-streaming)

                else:  # rag or search — streaming
                    status_box.write("⚙️ Streaming response...")
                    # Consume generator now (outside write_stream so we can
                    # collect tokens while the status box is still open)
                    chunks = []
                    for chunk in result["content"]:
                        chunks.append(chunk)
                    full_response = "".join(chunks)

                status_box.update(
                    label=STATUS_DONE.get(rtype, "✅ Done"),
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                err = str(e)
                status_box.update(label="⚠️ Error", state="error", expanded=True)
                if "429" in err or "rate_limit" in err.lower():
                    full_response = "⏳ Rate limit reached — please wait a moment and retry."
                elif "401" in err or "api_key" in err.lower():
                    full_response = "🔑 API key error — check your `.env` file."
                else:
                    full_response = f"⚠️ {err}"

        # ── Display response AFTER status closes ──────────────────────────────
        if response_is_img and result_data:
            st.image(result_data, caption="Generated by Pollinations AI · flux model")
            st.caption(f"Prompt: {result.get('prompt', user_input)}")
        elif full_response:
            st.markdown(full_response)

        # Agent badge
        try:
            badge_label = result["label"]
            st.markdown(
                f'<span class="agent-badge badge-{rtype}">{badge_label}</span>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    # Cache updated vector store
    if vs_ref[0] is not None:
        st.session_state.vector_store = vs_ref[0]

    # 4 — persist to session state + SQLite
    msg_record: dict = {
        "role":       "assistant",
        "content":    full_response,
        "badge":      result.get("label", ""),
        "badge_type": rtype,
    }
    if response_is_img and result_data:
        msg_record["is_image"] = True
        msg_record["content"]  = result_data
        msg_record["caption"]  = f"Prompt: {result.get('prompt', '')}"

    st.session_state.messages.append(msg_record)
    save_message("assistant", full_response if not response_is_img else "[Generated Image]")


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;color:#1E2A3A;font-size:0.65rem;
            border-top:1px solid rgba(0,229,255,0.05);
            padding:1.2rem 0 7rem;margin-top:2rem;
            font-family:JetBrains Mono,monospace;'>
  🐺 Wolf Scholar v2 &nbsp;·&nbsp; Groq Llama 3.3 70B &nbsp;·&nbsp;
  Llama-4 Scout &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Pollinations flux &nbsp;·&nbsp;
  LangSmith<br>
  <span style='color:rgba(0,229,255,0.2);'>
    Precision Intelligence for Academic &amp; Diagnostic Research
    &nbsp;·&nbsp; Engineered by Abdullah Ibne Tayeb Tamur
  </span>
</div>
""", unsafe_allow_html=True)
