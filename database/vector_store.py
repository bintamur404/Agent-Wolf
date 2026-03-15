"""
database/vector_store.py
PDF ingestion pipeline: load → chunk → embed → FAISS → retrieve.
Also owns all SQLite chat history persistence.
"""
import os
import sqlite3
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()


# ── Stable CPU embeddings ─────────────────────────────────────────────────────
# Avoids the HuggingFaceEmbeddings meta-tensor crash on Windows
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


_EMBEDDINGS = None  # lazy singleton per process


def _get_embeddings() -> _SafeEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = _SafeEmbeddings()
    return _EMBEDDINGS


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_vector_store(pdf_path: str) -> FAISS:
    """
    Ingest a PDF into an in-memory FAISS index.
    Rebuilt fresh for every PDF upload.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, _get_embeddings())


def retrieve_context(vector_store: FAISS, query: str, k: int = 3) -> str:
    """Return top-k similar chunks joined as a single context string."""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── SQLite chat history ───────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "wolf_scholar.db")


def init_db() -> None:
    """Create chat_history table if it doesn't already exist."""
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
    """Persist one chat turn to SQLite."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()


def load_history() -> list:
    """Return all persisted chat turns ordered by insertion."""
    with sqlite3.connect(_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id"
        ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]
