import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join("data", "wolf_history.db")

def init_db():
    """Initializes the SQLite database."""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_message(agent, role, content, metadata=None):
    """Saves a single chat message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (agent, role, content, metadata)
        VALUES (?, ?, ?, ?)
    """, (agent, role, content, json.dumps(metadata) if metadata else None))
    conn.commit()
    conn.close()

def load_history(agent, limit=50):
    """Loads chat history for a specific agent."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, metadata FROM chat_history 
        WHERE agent = ? 
        ORDER BY timestamp ASC LIMIT ?
    """, (agent, limit))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "role": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]) if row[2] else {}
        })
    return history

def clear_agent_history(agent):
    """Clears history for a specific agent."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE agent = ?", (agent,))
    conn.commit()
    conn.close()
