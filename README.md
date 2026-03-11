# 🐺 Agent Wolf: Elite Multi-Modal Research Assistant

Agent Wolf is a high-performance, multi-modal AI platform designed for technical research, document analysis, and visual intelligence. Built on the **Groq LPU Inference Engine** and **LangChain**, it fulfills all requirements for the advanced AI/ML research assignment.

## 🔗 Live Tracing Link
**Public LangSmith Trace**: [INSERT_YOUR_PUBLIC_LINK_HERE]
*(Generate this link by clicking "Share" on your successful trace in the LangSmith dashboard)*

## 🚀 Key Features & Compliance

### 1. The Three Specialized Agents
- **📚 Knowledge Core (RAG)**: Ingests PDFs/TXTs via `RecursiveCharacterTextSplitter`. Retrieves strictly **top 3 chunks (k=3)** for precision.
- **🌐 Global Intelligence (Search)**: Real-time internet scanning via DuckDuckGo with full source citations.
- **👁️ Vision Synthesis (OCR)**: Multi-modal analysis via **Llama 4 Scout** (17B) for visual data extraction.

### 2. Technical Architecture
- **LLM Engine**: Groq (Llama 3.3 70B & Llama 4 Scout).
- **Embeddings**: `all-MiniLM-L6-v2` (running locally on **CPU**).
- **Vector Storage**: **FAISS** (Local relational lookup).
- **History Database**: **SQLite** (`database/wolf_history.db`).
- **Observability**: Fully integrated with **LangSmith** for real-time telemetry.

### 3. Modular Directory Structure
```text
Agent-Wolf-Researcher/
├── app/                # UI Layer (Streamlit Frontend)
├── agents/             # Logic Layer (Search, RAG, Vision)
├── utils/              # Infra Layer (LLM, Database Utils)
├── database/           # Persistence Layer (SQLite & FAISS Index)
├── requirements.txt    # Dependency Graph
└── .env                # Secure Environment (Hidden in Git)
```

## 🛠️ Setup & Deployment

### Local Setup
1. Copy `.env.example` to `.env` and add your keys.
2. Ensure `.env` is the first line in your `.gitignore`.
3. Run the app:
   ```bash
   streamlit run app/app_ui.py
   ```

### ☁️ Streamlit Cloud Deployment
1. Push this repository to your GitHub.
2. In Streamlit Cloud, go to **Settings > Secrets**.
3. Inject your keys exactly as they appear in `.env`:
   ```toml
   GROQ_API_KEY = "gsk_..."
   LANGCHAIN_API_KEY = "lsv2_pt_..."
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_PROJECT = "Agent-Wolf"
   ```

---
*Developed for AI/ML Research Portfolio* 🐺✨
