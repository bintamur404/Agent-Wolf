# Wolf Scholar: Multi-Modal Research Copilot 🐺

> **Created by: Abdullah Ibne Tayeb Tamur**

Wolf Scholar is a unified, intelligent research platform. It features a Gemini-style "frictionless command center" that automatically routes your queries based on file attachments — whether it's academic search, document RAG, or vision analysis.

---

## 🔗 LangSmith Tracing (Live Observability)

This project is fully integrated with **LangSmith** for real-time agent telemetry and trace inspection.

- **Project Name:** `Agent-Wolf`
- **Tracing:** Enabled via `LANGCHAIN_TRACING_V2=true`
- **Endpoint:** `https://api.smith.langchain.com`
- **GitHub Repository:** [https://github.com/bintamur404/Agent-Wolf](https://github.com/bintamur404/Agent-Wolf)

> ℹ️ To view your own traces, sign up at [smith.langchain.com](https://smith.langchain.com) and add your `LANGCHAIN_API_KEY` to the `.env` file as described below.

---

## 🚀 How to Run the Project Locally

### 1. Create and Activate the Virtual Environment

To keep dependencies isolated, create a virtual environment (`venv`):

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Required Files to Create (`.env`)

For security, API keys are **not included** in this repository. You must create a `.env` file in the root directory (reference `.env.example`). It must contain:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="Agent-Wolf"
```

### 4. Run the Application
```bash
streamlit run app/app_ui.py
```

---

### ✨ Unified Intelligent Routing (The "Wolf Scholar" Engine)

The core innovation of this version is the **unified chat input**. Instead of manually switching modes, the platform detects your intent:
- **No Attachment** ➡️ **Global Search** (Web-grounded synthesis)
- **PDF/TXT Upload** ➡️ **Wolf Knowledge Base** (FAISS-backed RAG)
- **Image Upload** ➡️ **Vision Analysis** (Llama-4 Scout OCR)

---

## 🏗️ Architecture

```text
Wolf-Scholar/
├── app/                # UI Layer (Streamlit Frontend)
├── agents/             # Logic Layer (Search, RAG, Vision, Image Gen)
├── utils/              # Infra Layer (LLM factory, DB utils)
├── database/           # Persistence Layer (SQLite history + FAISS index)
├── requirements.txt    # Dependency graph
└── .env.example        # Template for environment variables
```

### The Specialized Research Stack

| Tool | Capability | Engine |
| --- | --- | --- |
| 🌐 Global Search | Real-time web search | DuckDuckGo + Llama 3.3 |
| 📚 Knowledge Base | PDF/TXT Intelligence | FAISS + RAG |
| 👁️ Vision | Image & Document OCR | Llama 4 Scout |
| 🔬 ArXiv Scholar | Deep academic search | ArXiv API |
| 🎨 Image Gen | Research visualization | Gemini 2.0 |

---

## 📝 Project Discussion

1. **Hybrid Retrieval:** I implemented FAISS for local document retrieval paired with SQLite for chat persistence to ensure maximum stability and speed during research.
2. **Resource Optimization:** Token management is handled by limiting FAISS retrieval to `k=3` chunks, preventing 429 errors while maintaining high accuracy.
3.  **Model Selection:** Utilizes latest high-speed models (Llama 3.3 70B & Llama 4 Scout) for professional-grade reasoning.

---

*Engineered & Deployed by **Abdullah Ibne Tayeb Tamur** — Multi-Modal AI Research Architecture* 🐺✨
