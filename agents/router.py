"""
agents/router.py
Wolf Scholar — four-route intelligence brain.

ROUTE A  PDF upload  → RAG via FAISS + Llama 3.3 70B  (streaming)
ROUTE B  Image upload → Vision via Llama-4 Scout       (full string)
ROUTE C  Text only    → Academic search DuckDuckGo     (streaming)
ROUTE D  Text + image keyword → Pollinations.AI image gen (bytes)

Public API:
    route(user_input, uploaded_file=None, vs_ref=None) → dict
    {
        "type":    "rag" | "vision" | "search" | "image",
        "content": generator | str | bytes,
        "label":   str,
        "prompt":  str   (only for type=="image")
    }
"""
import base64
import os
import random
import re
import tempfile
import urllib.parse
from typing import Generator

import requests as _requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage

from database.vector_store import build_vector_store, retrieve_context
from utils.llm_factory import get_text_llm, get_vision_llm

load_dotenv()

# ── Image-generation trigger keywords ─────────────────────────────────────────
_IMAGE_TRIGGERS = [
    "generate", "create image", "create an image", "draw",
    "visualize", "illustrate", "make an image", "make a picture",
    "show me a picture", "render", "depict", "paint", "sketch",
    "design an image", "generate an image", "generate a picture",
]

# ── System prompts (verbatim per spec) ────────────────────────────────────────
_RAG_SYSTEM = (
    "You are Wolf Scholar, a senior academic researcher specializing in deep learning "
    "architectures (dual-stream networks, FreqViT, ViT-based models) and computational "
    "pathology. Analyze the provided research context. Identify methodological approaches, "
    "datasets used, key results, and—most importantly—explicit research gaps or limitations "
    "the authors acknowledge. Structure your response with headers: "
    "## Summary, ## Methodology, ## Key Findings, ## Research Gaps."
)

_VISION_SYSTEM = (
    "You are Wolf Scholar's diagnostic vision module, acting as a board-certified "
    "pathologist and precision agriculture specialist. Examine the attached image "
    "meticulously. Identify: (1) visible anomalies, lesions, or disease markers, "
    "(2) affected tissue/crop regions with approximate severity, "
    "(3) likely differential diagnoses ranked by confidence, "
    "(4) recommended next diagnostic steps. Be precise, clinical, and evidence-based."
)

_SEARCH_SYSTEM = (
    "You are Wolf Scholar, an academic research assistant. You have been given live "
    "search results from arXiv, Nature, and IEEE. Synthesize these results into a "
    "coherent, well-cited academic summary. Always mention paper titles, authors if "
    "available, and publication venues. Highlight consensus findings and active debates."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_image_request(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _IMAGE_TRIGGERS)


def _extract_description(prompt: str) -> str:
    """Strip verb prefix ('generate a ...') to get the core image description."""
    clean = re.sub(
        r"^(please\s+)?(generate an?|create an? image of?|create an?|draw|"
        r"visualize|illustrate|make an? image of?|make an?|show me a picture of?|"
        r"render|depict|paint|sketch|design an? image of?|generate)\s+",
        "",
        prompt.strip(),
        flags=re.IGNORECASE,
    ).strip()
    return clean or prompt


# ── Route A — PDF / RAG ───────────────────────────────────────────────────────

def _route_rag(query: str, pdf_bytes: bytes, vs_ref: list) -> Generator:
    """Save PDF to tmp, build FAISS, retrieve context, stream LLM."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        vs = build_vector_store(tmp_path)
        vs_ref[0] = vs
        context = retrieve_context(vs, query)
    finally:
        os.unlink(tmp_path)

    llm = get_text_llm()
    msgs = [
        SystemMessage(content=_RAG_SYSTEM),
        HumanMessage(
            content=f"Research Context:\n{context}\n\nUser Question:\n{query}"
        ),
    ]
    return (chunk.content for chunk in llm.stream(msgs) if chunk.content)


# ── Route B — Image / Vision ──────────────────────────────────────────────────

def _route_vision(query: str, img_bytes: bytes, mime: str) -> str:
    """Base64-encode image, call vision LLM, return full response string."""
    b64 = base64.b64encode(img_bytes).decode()
    llm = get_vision_llm()
    msgs = [
        SystemMessage(content=_VISION_SYSTEM),
        HumanMessage(content=[
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": query},
        ]),
    ]
    return llm.invoke(msgs).content


# ── Route C — Academic Search ─────────────────────────────────────────────────

def _route_search(query: str) -> Generator:
    """DuckDuckGo → academic sites → stream LLM synthesis."""
    academic_q = f"{query} site:arxiv.org OR site:nature.com OR site:ieee.org"
    ddg = DuckDuckGoSearchRun()
    results = ddg.run(academic_q)
    llm = get_text_llm()
    msgs = [
        SystemMessage(content=_SEARCH_SYSTEM),
        HumanMessage(
            content=f"Search Results:\n{results}\n\nUser Query:\n{query}"
        ),
    ]
    return (chunk.content for chunk in llm.stream(msgs) if chunk.content)


# ── Route D — Image Generation (Hugging Face) ─────────────────────────────────

def _route_image_gen(prompt: str) -> tuple[bytes, str]:
    """
    Call Hugging Face FLUX.1-schnell model. Returns (image_bytes, clean_description).
    """
    description = _extract_description(prompt)
    token = os.getenv("HF_TOKEN")
    
    if not token:
        raise ValueError("HF_TOKEN is missing from your .env file.")

    url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": description}

    r = _requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise ValueError(f"Hugging Face API Error {r.status_code}: {r.text[:200]}")
        
    return r.content, description


# ── Public router ─────────────────────────────────────────────────────────────

def route(
    user_input: str,
    uploaded_file=None,
    vs_ref: list | None = None,
) -> dict:
    """
    Route the user turn to the correct agent.

    Returns:
        dict with keys: type, content, label, prompt (image only)
    """
    if vs_ref is None:
        vs_ref = [None]

    # ── File attached ──────────────────────────────────────────────────────────
    if uploaded_file is not None:
        mime = uploaded_file.type
        # Read bytes once; caller must have sought to 0 if file was previewed
        raw = uploaded_file.read()
        if not raw:
            raise ValueError(
                "File appears empty. If you previewed it above, this is a "
                "Streamlit seek bug — please refresh and try again."
            )

        if mime == "application/pdf":
            return {
                "type":    "rag",
                "content": _route_rag(user_input, raw, vs_ref),
                "label":   "📄 RAG · FAISS Knowledge Base",
            }

        if mime.startswith("image/"):
            return {
                "type":    "vision",
                "content": _route_vision(user_input, raw, mime),
                "label":   "🔬 Vision · Llama-4 Scout",
            }

        raise ValueError(f"Unsupported file type: {mime}")

    # ── No file — image generation keyword? ───────────────────────────────────
    if _is_image_request(user_input):
        img_bytes, clean_prompt = _route_image_gen(user_input)
        return {
            "type":    "image",
            "content": img_bytes,
            "label":   "🎨 Image Gen · Hugging Face FLUX",
            "prompt":  clean_prompt,
        }

    # ── Default: academic web search ──────────────────────────────────────────
    return {
        "type":    "search",
        "content": _route_search(user_input),
        "label":   "🔍 Search · arXiv · Nature · IEEE",
    }
