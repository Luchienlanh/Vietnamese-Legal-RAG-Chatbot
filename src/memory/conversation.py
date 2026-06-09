import json
import sqlite3
from datetime import datetime
from pathlib import Path


MEMORY_DB = Path("data/index/conversations.sqlite")


def get_connection():
    MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db():
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT PRIMARY KEY,
            active_context_json TEXT,
            last_evidence_json TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, turn_index)
        """
    )

    conn.commit()
    conn.close()


def next_turn_index(conn, session_id: str) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(turn_index), 0) AS max_turn
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    return int(row["max_turn"]) + 1


def load_memory(session_id: str, max_messages: int = 6) -> dict:
    init_memory_db()
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT role, content, turn_index, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY turn_index DESC, id DESC
        LIMIT ?
        """,
        (session_id, max_messages),
    ).fetchall()

    state = conn.execute(
        """
        SELECT active_context_json, last_evidence_json
        FROM session_state
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    conn.close()

    messages = [dict(row) for row in reversed(rows)]
    active_context = {}
    last_evidence = []

    if state:
        if state["active_context_json"]:
            active_context = json.loads(state["active_context_json"])

        if state["last_evidence_json"]:
            last_evidence = json.loads(state["last_evidence_json"])

    return {
        "session_id": session_id,
        "messages": messages,
        "active_context": active_context,
        "last_evidence": last_evidence,
    }


def compact_evidence(evidence, max_items: int = 8) -> list:
    output = []

    for item in evidence[:max_items]:
        metadata = item.get("metadata", {})

        output.append(
            {
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "source_url": item.get("source_url"),
                "context_mode": item.get("context_mode"),
                "metadata": {
                    "topic_title": metadata.get("topic_title"),
                    "subject_title": metadata.get("subject_title"),
                    "article_title": metadata.get("article_title"),
                    "original_article_number": metadata.get("original_article_number"),
                    "source_instrument_key": metadata.get("source_instrument_key"),
                    "doc_name": metadata.get("doc_name"),
                    "doc_code": metadata.get("doc_code"),
                    "subject": metadata.get("subject"),
                },
            }
        )

    return output


def build_active_context(search_result: dict) -> dict:
    route = search_result.get("route") or {}
    topic = route.get("topic") or {}
    subject = route.get("subject") or {}
    evidence = search_result.get("evidence") or []

    return {
        "topic_title": topic.get("topic_title"),
        "subject_title": subject.get("subject_title"),
        "recent_titles": [
            item.get("title")
            for item in evidence[:8]
            if item.get("title")
        ],
        "recent_source_urls": [
            item.get("source_url")
            for item in evidence[:8]
            if item.get("source_url")
        ],
    }


def save_turn(
    session_id: str,
    user_message: str,
    assistant_answer: str,
    search_result: dict,
) -> None:
    init_memory_db()
    conn = get_connection()

    now = datetime.utcnow().isoformat()
    user_turn = next_turn_index(conn, session_id)
    assistant_turn = user_turn + 1

    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, turn_index, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, "user", user_message, user_turn, now),
    )

    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, turn_index, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, "assistant", assistant_answer, assistant_turn, now),
    )

    active_context = build_active_context(search_result)
    last_evidence = compact_evidence(search_result.get("evidence") or [])

    conn.execute(
        """
        INSERT INTO session_state (
            session_id,
            active_context_json,
            last_evidence_json,
            updated_at
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            active_context_json = excluded.active_context_json,
            last_evidence_json = excluded.last_evidence_json,
            updated_at = excluded.updated_at
        """,
        (
            session_id,
            json.dumps(active_context, ensure_ascii=False),
            json.dumps(last_evidence, ensure_ascii=False),
            now,
        ),
    )

    conn.commit()
    conn.close()


def reset_session(session_id: str) -> None:
    init_memory_db()
    conn = get_connection()

    conn.execute(
        "DELETE FROM messages WHERE session_id = ?",
        (session_id,),
    )

    conn.execute(
        "DELETE FROM session_state WHERE session_id = ?",
        (session_id,),
    )

    conn.commit()
    conn.close()


init_db = init_memory_db
