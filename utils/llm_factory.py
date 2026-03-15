"""
utils/llm_factory.py
Centralized Groq LLM factory.  Zero hardcoded keys — all from .env.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_text_llm() -> ChatGroq:
    """
    Streaming LLM for text / reasoning tasks.
    Model : llama-3.3-70b-versatile
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        streaming=True,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def get_vision_llm() -> ChatGroq:
    """
    Vision LLM for multimodal / OCR tasks.
    Model : meta-llama/llama-4-scout-17b-16e-instruct
    """
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )
