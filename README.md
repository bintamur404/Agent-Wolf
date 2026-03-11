# 🐺 Agent Wolf: Elite Multi-Modal Research Assistant

Agent Wolf is a high-performance, multi-modal AI assistant designed for technical research, document analysis, and visual intelligence. Built on the **Groq LPU Inference Engine**, it offers lightning-fast response times for complex retrieval and vision tasks.

## 🚀 Key Features

- **🌐 Global Intelligence (Search)**: Real-time web scanning via DuckDuckGo and Llama-3.3.
- **📚 Knowledge Core (RAG)**: Advanced Document Q&A using FAISS vector embeddings and SQLite for persistent conversation history.
- **👁️ Vision Synthesis (OCR)**: Multi-modal image analysis powered by Llama 4 Scout.
- **📊 Observability**: Fully integrated with **LangSmith** for deep trace analysis and performance monitoring.
- **💎 Premium UI**: Glassmorphism dark-mode interface with a focus on user experience and responsiveness.

## 🛠️ Technical Stack

- **LLM Engine**: Groq (Llama 3.3 / Llama 4 Scout)
- **Orchestration**: LangChain
- **Databases**: FAISS (Vector) & SQLite (Relational)
- **Frontend**: Streamlit + Custom CSS
- **Observability**: LangSmith

## 📁 Project Structure

```text
simple-chatbot-main/
├── agents/             # Logic for Search, RAG, and Vision agents
├── utils/              # Shared utilities (DB, LLM, helper functions)
├── data/               # Persistent SQLite storage
├── app.py              # Main multi-modal entry point
└── requirements.txt    # Project dependencies
```

## ⚙️ Setup Instructions

1. **Clone & Install**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Copy `.env.example` to `.env` and add your API keys:
   ```env
   GROQ_API_KEY=gsk_...
   LANGCHAIN_API_KEY=ls__... (Optional but recommended)
   ```

3. **Launch**:
   ```bash
   streamlit run app.py
   ```

---
*Developed for AI/ML Research Portfolio* 🐺✨
