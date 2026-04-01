"""SQLite-backed avatar metadata store.

Stores avatar metadata (id, registration time, image path, size, clip status)
in a persistent SQLite database. The heavy feature data (source_info) is still
stored as pickle files on disk — SQLite stores only the metadata index.

Usage:
    db = AvatarDB("/workspace/avatar_cache/avatars.db")
    db.upsert("avatar123", image_path="/workspace/images/avatar123.png", size_mb=9.2)
    meta = db.get("avatar123")
    all_avatars = db.list_all(limit=50, offset=0)
"""

import sqlite3
import time
import threading
from pathlib import Path


class AvatarDB:
    """Thread-safe SQLite avatar metadata store."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS avatars (
                    avatar_id TEXT PRIMARY KEY,
                    image_path TEXT,
                    size_mb REAL DEFAULT 0,
                    registered_at REAL DEFAULT 0,
                    last_accessed_at REAL DEFAULT 0,
                    clips_ready INTEGER DEFAULT 0,
                    prerender_status TEXT DEFAULT 'none'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    avatar_id TEXT,
                    created_at REAL DEFAULT 0,
                    stopped_at REAL,
                    status TEXT DEFAULT 'active'
                )
            """)
            conn.commit()
            conn.close()

    def upsert(self, avatar_id: str, **kwargs):
        """Insert or update avatar metadata."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO avatars (avatar_id, registered_at, last_accessed_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(avatar_id) DO UPDATE SET last_accessed_at=?""",
                (avatar_id, now, now, now),
            )
            for key, value in kwargs.items():
                if key in ("image_path", "size_mb", "clips_ready", "prerender_status"):
                    conn.execute(f"UPDATE avatars SET {key}=? WHERE avatar_id=?", (value, avatar_id))
            conn.commit()
            conn.close()

    def get(self, avatar_id: str) -> dict | None:
        """Get avatar metadata."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM avatars WHERE avatar_id=?", (avatar_id,)).fetchone()
            conn.close()
            return dict(row) if row else None

    def list_all(self, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """List avatar metadata with pagination."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM avatars ORDER BY last_accessed_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM avatars").fetchone()[0]
            conn.close()
            return [dict(r) for r in rows], total

    def delete(self, avatar_id: str):
        """Delete avatar metadata."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM avatars WHERE avatar_id=?", (avatar_id,))
            conn.commit()
            conn.close()

    def log_session(self, session_id: str, avatar_id: str):
        """Log a new session."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, avatar_id, created_at, status) VALUES (?, ?, ?, 'active')",
                (session_id, avatar_id, time.time()),
            )
            conn.commit()
            conn.close()

    def end_session(self, session_id: str):
        """Mark session as stopped."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE sessions SET stopped_at=?, status='stopped' WHERE session_id=?",
                (time.time(), session_id),
            )
            conn.commit()
            conn.close()

    def session_stats(self) -> dict:
        """Get session statistics."""
        with self._lock:
            conn = self._get_conn()
            active = conn.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            conn.close()
            return {"active": active, "total": total}
