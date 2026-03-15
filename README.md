# Agent Wolf: Multi-Modal Research Copilot 🐺
This is a multi-agent system built for my AI/ML course. It features Web Search, Document Q&A (RAG via FAISS), and Vision/OCR capabilities utilizing Groq's high-speed Llama models and Streamlit.

🚀 How to Run the Project Locally
1. Create and Activate the Virtual Environment
To keep dependencies isolated, create a virtual environment (venv):

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

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Required Files to Create (.env)
For security, API keys are not included in this repository. You must create a .env file in the root directory (reference .env.example). It must contain:

- **GROQ_API_KEY**: Your Groq API key for Llama inference.
- **LANGCHAIN_TRACING_V2=true**: To enable LangSmith telemetry.
- **LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"**
- **LANGCHAIN_API_KEY**: Your LangSmith API key.
- **LANGCHAIN_PROJECT="Agent_Wolf"**

4. Run the Application
```bash
streamlit run app/app_ui.py
```

📝 Remaining Problems & Discussion Points
- **Vector Database Deployment**: I implemented FAISS for local document retrieval to avoid the sqlite-vec Streamlit Cloud compilation crashes. I used standard SQLite for relational chat history. Is this hybrid DB approach acceptable for the final grading rubric?
- **Model Deprecation Handling**: The originally planned `llama-3.2-11b-vision-preview` was decommissioned by Groq. I migrated the OCR agent to `llama-4-scout-17b-16e-instruct` (via Llama-4 Scout).
- **API Rate Limiting**: We need to discuss token management best practices (e.g., limiting FAISS retrieval to `k=3` chunks) to prevent hitting the 429 RESOURCE_EXHAUSTED Groq limit during intensive RAG testing.
