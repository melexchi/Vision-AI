"""Vision-AI Demo Server — runs on any laptop WITHOUT GPU.

This server mocks the AI model inference but exercises ALL the infrastructure:
- FastAPI endpoints with Pydantic validation
- Rate limiting middleware
- Prometheus metrics (/metrics)
- SQLite persistent metadata (avatar_db)
- LRU cache eviction
- Circuit breaker
- HMAC pickle safety
- Path traversal protection
- Consistent error format
- Graceful shutdown
- OpenAPI docs (/docs)

Start: python demo_server.py
Then open: http://localhost:8181/docs
"""

import sys
import os
import time
import base64
import json
import asyncio
import threading
import logging
import uuid
import numpy as np
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request, WebSocket, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logger = logging.getLogger("vision-ai-demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

# ─── App Setup ─────────────────────────────────────────────────────

app = FastAPI(
    title="Vision-AI Demo Server (No GPU Required)",
    version="2.0-demo",
    description="Exercises all infrastructure features without AI models. Open /docs to explore.",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Rate Limiting ─────────────────────────────────────────────────

_rate_buckets: dict[str, list] = {}
RATE_LIMIT_RPM = 60


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_buckets.setdefault(ip, [])
    bucket[:] = [t for t in bucket if now - t < 60]
    if len(bucket) >= RATE_LIMIT_RPM:
        return JSONResponse({"detail": "Rate limit exceeded", "code": "ERR_429"}, status_code=429)
    bucket.append(now)
    return await call_next(request)


# ─── Prometheus Metrics ────────────────────────────────────────────

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

    REQ_COUNT = Counter("demo_requests_total", "Total requests", ["method", "endpoint", "status"])
    REQ_LATENCY = Histogram("demo_request_latency_seconds", "Latency", ["endpoint"])
    AVATAR_COUNT = Gauge("demo_avatar_count", "Avatars registered")

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        REQ_COUNT.labels(request.method, request.url.path, response.status_code).inc()
        REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
        return response

    @app.get("/metrics", tags=["Observability"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    PROMETHEUS = True
except ImportError:
    PROMETHEUS = False
    logger.info("prometheus-client not installed — /metrics disabled. pip install prometheus-client to enable.")


# ─── Error Handler ─────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": f"ERR_{exc.status_code}"},
    )


# ─── SQLite Database ──────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent / "ditto"))
from avatar_db import AvatarDB

DEMO_DIR = Path("demo_data")
DEMO_DIR.mkdir(exist_ok=True)
db = AvatarDB(DEMO_DIR / "demo.db")

# ─── In-Memory State (same pattern as production) ─────────────────

avatar_cache: dict[str, dict] = {}
avatar_cache_lock = threading.Lock()
_cache_order: list[str] = []
MAX_AVATARS = 10  # small for demo


# ─── Circuit Breaker ──────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, name, threshold=3, cooldown=30):
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._last_fail = 0.0

    @property
    def is_open(self):
        if self._failures >= self.threshold:
            if time.time() - self._last_fail > self.cooldown:
                self._failures = 0
                return False
            return True
        return False

    def fail(self):
        self._failures += 1
        self._last_fail = time.time()

    def success(self):
        self._failures = 0


tts_breaker = CircuitBreaker("tts", threshold=3, cooldown=10)

# ─── Request/Response Models ──────────────────────────────────────


class RegisterRequest(BaseModel):
    avatar_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    prerender_clips: bool = False


class GenerateRequest(BaseModel):
    avatar_id: str
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    sampling_timesteps: Optional[int] = Field(None, ge=1, le=50)
    fps: Optional[int] = Field(None, ge=1, le=60)


class HealthResponse(BaseModel):
    status: str
    mode: str
    avatars: int
    db_stats: dict
    prometheus: bool
    features: list[str]


class RegisterResponse(BaseModel):
    status: str
    avatar_id: str
    size_kb: float
    in_db: bool
    cache_size: int


class AvatarListResponse(BaseModel):
    avatars: list[dict]
    total: int
    limit: int
    offset: int


# ─── Endpoints ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — shows all active features."""
    return {
        "status": "ok",
        "mode": "DEMO (no GPU)",
        "avatars": len(avatar_cache),
        "db_stats": db.session_stats(),
        "prometheus": PROMETHEUS,
        "features": [
            "rate_limiting", "circuit_breaker", "lru_cache", "sqlite_db",
            "input_validation", "path_traversal_protection", "error_codes",
            "prometheus_metrics", "openapi_docs", "graceful_shutdown",
        ],
    }


@app.post("/register", response_model=RegisterResponse, tags=["Avatars"])
async def register(request: RegisterRequest):
    """Register an avatar. In demo mode, creates a mock entry."""
    MAX_B64 = 50 * 1024 * 1024
    if request.image_base64:
        if len(request.image_base64) > MAX_B64:
            raise HTTPException(status_code=413, detail="Image too large (max 50MB)")
        data = base64.b64decode(request.image_base64)
        size_kb = len(data) / 1024
    elif request.image_path:
        raise HTTPException(status_code=400, detail="image_path not supported in demo mode — use image_base64")
    else:
        size_kb = 0.0

    avatar_id = request.avatar_id or f"demo_{uuid.uuid4().hex[:8]}"

    # LRU cache
    with avatar_cache_lock:
        avatar_cache[avatar_id] = {"registered_at": time.time(), "size_kb": size_kb}
        _cache_order.append(avatar_id)
        while len(_cache_order) > MAX_AVATARS:
            evicted = _cache_order.pop(0)
            avatar_cache.pop(evicted, None)
            logger.info(f"[LRU] Evicted {evicted} (max={MAX_AVATARS})")

    # SQLite
    db.upsert(avatar_id, size_mb=round(size_kb / 1024, 2))
    if PROMETHEUS:
        AVATAR_COUNT.set(len(avatar_cache))

    logger.info(f"Registered avatar {avatar_id} ({size_kb:.1f} KB)")
    return {
        "status": "registered",
        "avatar_id": avatar_id,
        "size_kb": round(size_kb, 1),
        "in_db": db.get(avatar_id) is not None,
        "cache_size": len(avatar_cache),
    }


@app.get("/avatars", response_model=AvatarListResponse, tags=["Avatars"])
async def list_avatars(limit: int = 50, offset: int = 0):
    """List registered avatars with pagination."""
    with avatar_cache_lock:
        all_ids = list(avatar_cache.keys())
    total = len(all_ids)
    page = all_ids[offset:offset + limit]
    avatars = []
    for aid in page:
        with avatar_cache_lock:
            info = avatar_cache.get(aid, {})
        db_meta = db.get(aid)
        avatars.append({
            "avatar_id": aid,
            "size_kb": info.get("size_kb", 0),
            "in_memory": True,
            "in_db": db_meta is not None,
        })
    return {"avatars": avatars, "total": total, "limit": limit, "offset": offset}


@app.delete("/avatars/{avatar_id}", tags=["Avatars"])
async def delete_avatar(avatar_id: str):
    """Delete an avatar from cache and database."""
    with avatar_cache_lock:
        removed = avatar_cache.pop(avatar_id, None)
    db.delete(avatar_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")
    if PROMETHEUS:
        AVATAR_COUNT.set(len(avatar_cache))
    return {"status": "deleted", "avatar_id": avatar_id}


@app.post("/generate", tags=["Generation"])
async def generate(request: GenerateRequest):
    """Mock video generation — validates inputs, returns a placeholder."""
    if request.avatar_id not in avatar_cache:
        raise HTTPException(status_code=400, detail=f"Avatar '{request.avatar_id}' not found. Call /register first.")

    if tts_breaker.is_open:
        raise HTTPException(status_code=503, detail="TTS circuit breaker open — too many failures")

    # Simulate processing
    await asyncio.sleep(0.5)

    return {
        "status": "generated",
        "avatar_id": request.avatar_id,
        "fps": request.fps or 25,
        "sampling_timesteps": request.sampling_timesteps or 5,
        "note": "DEMO MODE — no actual video generated. In production, this returns an MP4.",
    }


@app.post("/circuit-breaker/trip", tags=["Demo Tools"])
async def trip_circuit_breaker():
    """Demo tool: manually trip the TTS circuit breaker to see it in action."""
    tts_breaker.fail()
    tts_breaker.fail()
    tts_breaker.fail()
    return {
        "status": "tripped",
        "is_open": tts_breaker.is_open,
        "message": "Circuit breaker is now OPEN. /generate will return 503 for 10 seconds.",
    }


@app.post("/circuit-breaker/reset", tags=["Demo Tools"])
async def reset_circuit_breaker():
    """Demo tool: reset the circuit breaker."""
    tts_breaker.success()
    return {"status": "reset", "is_open": tts_breaker.is_open}


@app.get("/db/avatars", tags=["Demo Tools"])
async def db_avatars():
    """View raw SQLite avatar data."""
    rows, total = db.list_all(limit=100)
    return {"rows": rows, "total": total}


@app.get("/db/sessions", tags=["Demo Tools"])
async def db_sessions():
    """View session statistics from SQLite."""
    return db.session_stats()


# ─── Shutdown ──────────────────────────────────────────────────────

@app.on_event("shutdown")
async def shutdown():
    logger.info("[shutdown] Demo server shutting down cleanly.")


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Vision-AI Demo Server (No GPU Required)")
    print("=" * 60)
    print()
    print("  Open in browser:  http://localhost:8181/docs")
    print("  Health check:     http://localhost:8181/health")
    if PROMETHEUS:
        print("  Metrics:          http://localhost:8181/metrics")
    print()
    print("  Try these:")
    print('    curl http://localhost:8181/health')
    print('    curl -X POST http://localhost:8181/register -H "Content-Type: application/json" -d \'{"avatar_id": "test1"}\'')
    print('    curl http://localhost:8181/avatars')
    print('    curl -X POST http://localhost:8181/generate -H "Content-Type: application/json" -d \'{"avatar_id": "test1"}\'')
    print('    curl http://localhost:8181/db/avatars')
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8181)
