from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith Tracing Configuration (Set early for LangChain)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Agent-Wolf")


def get_llm():
    """Returns a ChatGroq instance with tracing configured."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2048,
    )
