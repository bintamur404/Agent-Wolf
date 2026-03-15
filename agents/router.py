"""
agents/router.py
Four-route intelligence brain for Wolf Scholar.

A — PDF uploaded        → RAG Agent (FAISS + Groq)
B — Image uploaded      → Vision Agent (Groq Llama-4 Scout)
C — No file, text query → Academic Search (DuckDuckGo → arXiv/Nature/IEEE)
D — No file, image kw   → Image Generation (Pollinations.AI, free, no key)
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

# ── Image-generation trigger keywords ────────────────────────────────────────
IMAGE_TRIGGERS = [
    "generate", "create image", "draw", "visualize",
    "illustrate", "make an image", "show me a picture",
    "render", "depict", "paint", "sketch", "design an image",
]


# ── System Prompts ────────────────────────────────────────────────────────────
_RAG_SYSTEM = (
    "You are Wolf Scholar, a senior academic researcher specializing in deep learning "
    "architectures (dual-stream networks, FreqViT, ViT-based models) and computational "
    "pathology. Analyze the provided research context. Identify methodological approaches, "
    "datasets used, key results, and—most importantly—explicit research gaps or limitations "
    "the authors acknowledge. Structure your response with headers: "
    "## Summary, ## Methodology, ## Key Findings, ## Research Gaps."
)

_VISION_SYSTEM = (
    "You are Wolf Scholar's diagnostic vision module, acting as a board-certified pathologist "
    "and precision agriculture specialist. Examine the attached image meticulously. Identify: "
    "(1) visible anomalies, lesions, or disease markers, "
    "(2) affected tissue/crop regions with approximate severity, "
    "(3) likely differential diagnoses ranked by confidence, "
    "(4) recommended next diagnostic steps. Be precise, clinical, and evidence-based."
)

_SEARCH_SYSTEM = (
    "You are Wolf Scholar, an academic research assistant. You have been given live search "
    "results from arXiv, Nature, and IEEE. Synthesize these results into a coherent, "
    "well-cited academic summary. Always mention paper titles, authors if available, and "
    "publication venues. Highlight consensus findings and areas of active debate."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_image_request(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in IMAGE_TRIGGERS)


def _extract_image_description(prompt: str) -> str:
    """Strip leading generate/create/draw verbs to get the core description."""
    clean = re.sub(
        r"^(please\s+)?(generate|create|draw|visualize|illustrate|"
        r"make an image of|show me a picture of|render|depict|paint|sketch|"
        r"design an image of|make|create an image of)\s+",
        "",
        prompt.strip(),
        flags=re.IGNORECASE,
    ).strip()
    return clean or prompt


# ── Route A — RAG ─────────────────────────────────────────────────────────────
def _route_pdf(query: str, pdf_bytes: bytes, vs_ref: list) -> Generator:
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
        HumanMessage(content=f"Research Context:\n{context}\n\nUser Question:\n{query}"),
    ]
    return (chunk.content for chunk in llm.stream(msgs) if chunk.content)


# ── Route B — Vision ──────────────────────────────────────────────────────────
def _route_vision(query: str, image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    llm = get_vision_llm()
    msgs = [
        SystemMessage(content=_VISION_SYSTEM),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": query},
        ]),
    ]
    return llm.invoke(msgs).content


# ── Route C — Academic Search ─────────────────────────────────────────────────
def _route_search(query: str) -> Generator:
    academic_query = f"{query} site:arxiv.org OR site:nature.com OR site:ieee.org"
    ddg = DuckDuckGoSearchRun()
    results = ddg.run(academic_query)
    llm = get_text_llm()
    msgs = [
        SystemMessage(content=_SEARCH_SYSTEM),
        HumanMessage(content=f"Search Results:\n{results}\n\nUser Query:\n{query}"),
    ]
    return (chunk.content for chunk in llm.stream(msgs) if chunk.content)


# ── Route D — Image Generation (Pollinations.AI) ──────────────────────────────
def _route_image_gen(prompt: str) -> bytes:
    description = _extract_image_description(prompt)
    encoded = urllib.parse.quote(description)
    seed = random.randint(1, 99999)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1024&height=768&model=flux&nologo=true&seed={seed}"
    )
    resp = _requests.get(url, timeout=90)
    resp.raise_for_status()
    return resp.content, description


# ── Public router ─────────────────────────────────────────────────────────────
def route(user_input: str, uploaded_file=None, vs_ref: list = None) -> dict:
    """
    Route the user turn to the correct agent.

    Returns a dict:
      {"type": "rag|vision|search|image", "content": generator_or_bytes,
       "label": str, "prompt": str (image only)}
    """
    if vs_ref is None:
        vs_ref = [None]

    # ── File attached ──────────────────────────────────────────────────────────
    if uploaded_file is not None:
        mime = uploaded_file.type
        raw = uploaded_file.read()

        if mime == "application/pdf":
            return {
                "type":    "rag",
                "content": _route_pdf(user_input, raw, vs_ref),
                "label":   "📄 RAG · FAISS Knowledge Base",
            }

        if mime.startswith("image/"):
            return {
                "type":    "vision",
                "content": _route_vision(user_input, raw, mime),
                "label":   "🔬 Vision · Llama-4 Scout",
            }

        raise ValueError(f"Unsupported file type: {mime}")

    # ── No file — check for image-gen keywords ─────────────────────────────────
    if _is_image_request(user_input):
        img_bytes, clean_prompt = _route_image_gen(user_input)
        return {
            "type":    "image",
            "content": img_bytes,
            "label":   "🎨 Image Gen · Pollinations flux",
            "prompt":  clean_prompt,
        }

    # ── Default: academic search ───────────────────────────────────────────────
    return {
        "type":    "search",
        "content": _route_search(user_input),
        "label":   "🔍 Search · arXiv · Nature · IEEE",
    }
