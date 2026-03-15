"""
database/vector_store.py
PDF ingestion (FAISS) + SQLite chat history.

All public functions:
  build_vector_store(pdf_path)      → FAISS
  retrieve_context(vs, query, k=3) → str
  init_db()
  save_message(role, content)
  load_history()                    → list[dict]
"""
import os
import sqlite3
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()


# ── Safe CPU embeddings (avoids meta-tensor crash on Windows) ─────────────────
class _SafeEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._model.to("cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


_EMBEDDINGS: _SafeEmbeddings | None = None  # lazy singleton


def _get_embeddings() -> _SafeEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = _SafeEmbeddings()
    return _EMBEDDINGS


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_vector_store(pdf_path: str) -> FAISS:
    """Load a PDF from disk, chunk it, embed, return FAISS index."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, _get_embeddings())


def retrieve_context(vector_store: FAISS, query: str, k: int = 3) -> str:
    """Return top-k relevant chunks joined as a single string."""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ── SQLite ────────────────────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "wolf_scholar.db")


def init_db() -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()


def save_message(role: str, content: str) -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content, timestamp) VALUES (?,?,?)",
            (role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()


def load_history() -> list:
    with sqlite3.connect(_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id"
        ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


def clear_history() -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DELETE FROM chat_history")
        conn.commit()
