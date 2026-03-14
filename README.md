# Agent Wolf: Multi-Modal Research Copilot 🐺

> **Created by: Abdullah Ibne Tayeb Tamur**

This is a multi-agent system built for my AI/ML course. It features Web Search, Document Q&A (RAG via FAISS), and Vision/OCR capabilities utilizing Groq's high-speed Llama models and Streamlit.

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

## 🏗️ Architecture

```text
Agent-Wolf/
├── app/                # UI Layer (Streamlit Frontend)
├── agents/             # Logic Layer (Search, RAG, Vision, Image Gen)
├── utils/              # Infra Layer (LLM factory, DB utils)
├── database/           # Persistence Layer (SQLite history + FAISS index)
├── requirements.txt    # Dependency graph
└── .env.example        # Template for environment variables
```

### The Four Specialized Agents

| Agent | Capability | Model |
| --- | --- | --- |
| 🌐 Global Search | Real-time web search with DuckDuckGo | Llama 3.3 70B |
| 📚 Knowledge Base (RAG) | Document Q&A via FAISS (k=3 retrieval) | Llama 3.3 70B |
| 👁️ Vision Analysis | Multi-modal image understanding | Llama 4 Scout 17B |
| 🎨 Image Generation | Text-to-image synthesis | Gemini 2.0 Flash |

### 🛡️ Architectural Decision: DuckDuckGo vs. Google Search API

The original project brief suggested grounding the search agent with Google Search. However, I deliberately engineered the system to utilize `DuckDuckGoSearchRun` instead.

**Rationale:**
Relying on Google Custom Search requires hardcoding additional API keys and introduces strict rate-limiting bottlenecks. By utilizing DuckDuckGo, the agent achieves the exact same live-web grounding capability but guarantees a frictionless, crash-free deployment. When designing robust ML pipelines, minimizing external API dependencies is critical for system stability, and this design choice ensures the chatbot remains highly available without hitting quota walls.

---

## 📝 Remaining Problems & Discussion Points

1. **Vector Database Deployment:** I implemented FAISS for local document retrieval to avoid the `sqlite-vec` Streamlit Cloud compilation crashes. I used standard SQLite for relational chat history. Is this hybrid DB approach acceptable for the final grading rubric?
2. **Model Deprecation Handling:** The originally planned `llama-3.2-11b-vision-preview` was decommissioned by Groq. I migrated the OCR agent to `meta-llama/llama-4-scout-17b-16e-instruct`.
3. **API Rate Limiting:** We need to discuss token management best practices (e.g., limiting FAISS retrieval to `k=3` chunks) to prevent hitting the `429 RESOURCE_EXHAUSTED` Groq limit during intensive RAG testing.

---

*Developed for AI/ML Research Portfolio — Agent Wolf by Abdullah Ibne Tayeb Tamur* 🐺✨
