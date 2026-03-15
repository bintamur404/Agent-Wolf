"""
utils/llm_factory.py
Centralized factory for all Groq LLM clients.
Zero hardcoded keys — all from .env via python-dotenv.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_text_llm() -> ChatGroq:
    """
    Returns a streaming Groq client for text & reasoning tasks.
    Model: llama-3.3-70b-versatile
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        streaming=True,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def get_vision_llm() -> ChatGroq:
    """
    Returns a Groq client optimized for multimodal vision/OCR tasks.
    Model: meta-llama/llama-4-scout-17b-16e-instruct
    """
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )
