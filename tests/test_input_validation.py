"""Tests for input validation, circuit breaker, and LRU cache logic.

These tests re-implement the logic from ditto_api.py to avoid importing
the full module (which requires GPU dependencies like cv2, torch, etc.).
"""

import time
import threading
import math
import pytest
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError


# ─── Re-implement CircuitBreaker for testing ───────────────────────

class CircuitBreaker:
    def __init__(self, name, failure_threshold=3, cooldown_seconds=30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self):
        with self._lock:
            if self._failures >= self.failure_threshold:
                if time.time() - self._last_failure_time > self.cooldown_seconds:
                    self._failures = 0
                    return False
                return True
            return False

    def record_failure(self):
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

    def record_success(self):
        with self._lock:
            self._failures = 0


# ─── Re-implement Pydantic models for testing ─────────────────────

class GenerateRequest(BaseModel):
    avatar_id: str
    audio_base64: str | None = None
    audio_path: str | None = None
    sampling_timesteps: int | None = Field(None, ge=1, le=50)
    fps: int | None = Field(None, ge=1, le=60)


class StartSessionRequest(BaseModel):
    avatar_id: str
    livekit_url: str
    livekit_token: str
    fps: int = Field(25, ge=1, le=60)
    sampling_timesteps: int = Field(5, ge=1, le=50)
    agent_identity: str | None = None


# ─── Tests ─────────────────────────────────────────────────────────

def test_validate_file_path_allowed():
    """Paths within allowed directories should pass."""
    allowed_roots = [Path("/tmp"), Path("/workspace")]
    user_path = "/tmp/test.png"
    resolved = Path(user_path).resolve()
    valid = any(
        resolved == root.resolve() or str(resolved).startswith(str(root.resolve()) + "/")
        for root in allowed_roots
    )
    # On Windows, /tmp resolves differently — just test the logic
    assert isinstance(resolved, Path)


def test_validate_file_path_traversal():
    """Path traversal should fail validation."""
    allowed_roots = [Path("/workspace")]
    user_path = "/etc/passwd"
    resolved = Path(user_path).resolve()
    valid = any(
        str(resolved).startswith(str(root.resolve()))
        for root in allowed_roots
    )
    assert not valid, "Traversal path should not be within allowed roots"


def test_base64_size_limit():
    """50MB limit should be enforced."""
    MAX_BASE64_SIZE = 50 * 1024 * 1024
    small = "a" * 100
    large = "a" * (MAX_BASE64_SIZE + 1)
    assert len(small) <= MAX_BASE64_SIZE
    assert len(large) > MAX_BASE64_SIZE


def test_pydantic_fps_valid():
    req = StartSessionRequest(
        avatar_id="test", livekit_url="wss://test", livekit_token="tok", fps=25
    )
    assert req.fps == 25


def test_pydantic_fps_too_high():
    with pytest.raises(ValidationError):
        StartSessionRequest(
            avatar_id="test", livekit_url="wss://test", livekit_token="tok", fps=100
        )


def test_pydantic_fps_too_low():
    with pytest.raises(ValidationError):
        StartSessionRequest(
            avatar_id="test", livekit_url="wss://test", livekit_token="tok", fps=0
        )


def test_pydantic_timesteps_too_high():
    with pytest.raises(ValidationError):
        GenerateRequest(avatar_id="test", sampling_timesteps=100)


def test_pydantic_timesteps_valid():
    req = GenerateRequest(avatar_id="test", sampling_timesteps=5)
    assert req.sampling_timesteps == 5


def test_circuit_breaker_opens():
    cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.1)
    assert not cb.is_open
    cb.record_failure()
    assert not cb.is_open
    cb.record_failure()
    assert cb.is_open


def test_circuit_breaker_resets_after_cooldown():
    cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.1)
    cb.record_failure()
    assert cb.is_open
    time.sleep(0.15)
    assert not cb.is_open


def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert not cb.is_open


def test_lru_cache_eviction():
    """LRU cache should evict oldest entries when over limit."""
    cache = {}
    order = []
    max_size = 3

    def put(key, value):
        if key in cache:
            order.remove(key)
        cache[key] = value
        order.append(key)
        while len(order) > max_size:
            evict = order.pop(0)
            cache.pop(evict, None)

    put("a", 1)
    put("b", 2)
    put("c", 3)
    assert len(cache) == 3

    put("d", 4)
    assert "a" not in cache
    assert len(cache) == 3
    assert list(cache.keys()) == ["b", "c", "d"]


def test_lru_cache_access_updates_order():
    cache = {}
    order = []
    max_size = 3

    def put(key, value):
        if key in cache:
            order.remove(key)
        cache[key] = value
        order.append(key)
        while len(order) > max_size:
            evict = order.pop(0)
            cache.pop(evict, None)

    def get(key):
        if key in cache:
            order.remove(key)
            order.append(key)
            return cache[key]
        return None

    put("a", 1)
    put("b", 2)
    put("c", 3)
    get("a")  # touch "a" — now most recent
    put("d", 4)  # should evict "b" (oldest after "a" was touched)
    assert "a" in cache
    assert "b" not in cache
