"""Tests for SQLite avatar metadata store."""

import tempfile
import os
import pytest
from pathlib import Path

# Add ditto to path so we can import avatar_db
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ditto"))

from avatar_db import AvatarDB


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    return AvatarDB(tmp_path / "test.db")


def test_upsert_and_get(db):
    db.upsert("avatar1", image_path="/tmp/test.png", size_mb=9.5)
    meta = db.get("avatar1")
    assert meta is not None
    assert meta["avatar_id"] == "avatar1"
    assert meta["image_path"] == "/tmp/test.png"
    assert meta["size_mb"] == 9.5


def test_get_nonexistent(db):
    assert db.get("nonexistent") is None


def test_list_all(db):
    db.upsert("a1")
    db.upsert("a2")
    db.upsert("a3")
    rows, total = db.list_all(limit=2, offset=0)
    assert total == 3
    assert len(rows) == 2


def test_list_pagination(db):
    for i in range(10):
        db.upsert(f"avatar_{i}")
    rows, total = db.list_all(limit=3, offset=5)
    assert total == 10
    assert len(rows) == 3


def test_delete(db):
    db.upsert("to_delete")
    assert db.get("to_delete") is not None
    db.delete("to_delete")
    assert db.get("to_delete") is None


def test_session_logging(db):
    db.log_session("sess1", "avatar1")
    stats = db.session_stats()
    assert stats["active"] == 1
    assert stats["total"] == 1

    db.end_session("sess1")
    stats = db.session_stats()
    assert stats["active"] == 0
    assert stats["total"] == 1


def test_upsert_updates_existing(db):
    db.upsert("avatar1", size_mb=5.0)
    db.upsert("avatar1", size_mb=10.0)
    meta = db.get("avatar1")
    assert meta["size_mb"] == 10.0


def test_concurrent_access(db):
    """Verify thread safety with concurrent writes."""
    import threading

    def write_batch(prefix):
        for i in range(20):
            db.upsert(f"{prefix}_{i}", size_mb=float(i))

    threads = [threading.Thread(target=write_batch, args=(f"t{t}",)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _, total = db.list_all(limit=200)
    assert total == 100  # 5 threads × 20 avatars
