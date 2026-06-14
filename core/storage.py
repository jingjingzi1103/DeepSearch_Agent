"""SQLite 与会话级 FAISS 路径管理（纯 Python，可独立测试）。"""

import json
import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.db")
FAISS_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index_store")


def get_faiss_dir_for_conversation(conversation_id: int) -> str:
    return os.path.join(FAISS_ROOT_DIR, f"conv_{int(conversation_id)}")


def delete_faiss_dir_for_conversation(conversation_id: int) -> None:
    path = get_faiss_dir_for_conversation(conversation_id)
    if os.path.isdir(path):
        shutil.rmtree(path)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("PRAGMA table_info(messages)")
        cols = {row[1] for row in cur.fetchall()}
        if "conversation_id" not in cols:
            cur.execute("ALTER TABLE messages ADD COLUMN conversation_id INTEGER")
        if "trace_metadata" not in cols:
            cur.execute("ALTER TABLE messages ADD COLUMN trace_metadata TEXT")
        cur.execute("SELECT COUNT(1) FROM conversations")
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO conversations (title) VALUES (?)", ("默认会话",))
        cur.execute("UPDATE messages SET conversation_id = 1 WHERE conversation_id IS NULL")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id)"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                faithfulness REAL,
                relevance REAL,
                has_citation INTEGER,
                overall REAL,
                reason TEXT,
                judge_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_conversation(title: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def save_message_to_db(
    conversation_id: int,
    role: str,
    content: str,
    trace_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        meta_json = json.dumps(trace_metadata, ensure_ascii=False) if trace_metadata else None
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content, trace_metadata) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, meta_json),
        )
        conn.commit()
    finally:
        conn.close()


def load_history_from_db(conversation_id: int) -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, trace_metadata FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for role, content, trace_metadata in rows:
        item: Dict[str, Any] = {"role": role, "content": content}
        if trace_metadata:
            try:
                item["metadata"] = json.loads(trace_metadata)
            except json.JSONDecodeError:
                item["metadata"] = {}
        out.append(item)
    return out


def delete_conversation(conversation_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (int(conversation_id),))
        cur.execute("DELETE FROM evaluations WHERE conversation_id = ?", (int(conversation_id),))
        cur.execute("DELETE FROM conversations WHERE id = ?", (int(conversation_id),))
        conn.commit()
    finally:
        conn.close()
    delete_faiss_dir_for_conversation(conversation_id)


def list_conversations_with_preview() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                c.id,
                c.title,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS last_message
            FROM conversations c
            ORDER BY c.id DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for cid, title, last_msg in rows:
        preview = (last_msg or "").replace("\n", " ").strip()
        if len(preview) > 30:
            preview = preview[:30] + "..."
        out.append({"id": int(cid), "title": title, "preview": preview})
    return out


def update_conversation_title(conversation_id: int, new_title: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (new_title, int(conversation_id)),
        )
        conn.commit()
    finally:
        conn.close()


def clear_all_history() -> None:
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM evaluations")
        cur.execute("DELETE FROM conversations")
        conn.commit()
    finally:
        conn.close()
    if os.path.isdir(FAISS_ROOT_DIR):
        shutil.rmtree(FAISS_ROOT_DIR)


def save_evaluation_to_db(
    conversation_id: int,
    question: str,
    answer: str,
    eval_result: Dict[str, Any],
) -> None:
    """保存 Judge 评测结果。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO evaluations (
                conversation_id, question, answer,
                faithfulness, relevance, has_citation, overall, reason, judge_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(conversation_id),
                question,
                answer,
                float(eval_result.get("faithfulness", 0)),
                float(eval_result.get("relevance", 0)),
                1 if eval_result.get("has_citation") else 0,
                float(eval_result.get("overall", 0)),
                str(eval_result.get("reason", "")),
                str(eval_result.get("judge_model", "")),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_evaluations_for_conversation(conversation_id: int) -> List[Dict[str, Any]]:
    """加载某会话的全部 Judge 评分，按时间顺序返回。"""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT question, answer, faithfulness, relevance, has_citation,
                   overall, reason, judge_model, created_at
            FROM evaluations
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (int(conversation_id),),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "question": q,
            "answer": a,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "has_citation": bool(has_citation),
            "overall": overall,
            "reason": reason or "",
            "judge_model": judge_model or "",
            "created_at": created_at,
        }
        for q, a, faithfulness, relevance, has_citation, overall, reason, judge_model, created_at in rows
    ]
