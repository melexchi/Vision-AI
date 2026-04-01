# Feature Tracker

Recommended features from full codebase scan (2026-04-01). Delete this file when all items are resolved.

## P0 — Must Have Before Production (2)

- [x] **FEAT-001** Thread-safe state management — DONE 2026-04-01
  - Implemented as BUG-001/002/003 fix: threading.Lock on all three global dicts

- [x] **FEAT-002** Input validation layer — DONE 2026-04-01
  - Implemented as BUG-009/010/036/042: path traversal, base64 limits, Pydantic Field validators

## P1 — High Priority (4)

- [x] **FEAT-003** Structured logging — DONE 2026-04-01
  - Implemented as BUG-031: Python logging module in ditto + chatterbox with timestamps

- [x] **FEAT-004** Memory management with LRU eviction — DONE 2026-04-01
  - Avatar cache LRU (MAX_CACHED_AVATARS=50), clip cache LRU (CLIP_CACHE_MAX=20)
  - Session TTL (SESSION_TTL_SECONDS=1800), prerender job cleanup (1hr expiry)

- [x] **FEAT-005** Graceful shutdown — DONE 2026-04-01
  - Implemented as BUG-015: @app.on_event("shutdown") stops sessions + executor

- [x] **FEAT-006** Configuration management — DONE 2026-04-01
  - .env.example created, _validate_config() at startup, all paths/ports configurable

## P2 — Medium Priority (3)

- [x] **FEAT-007** Rate limiting — DONE 2026-04-01
  - Per-IP token bucket middleware (RATE_LIMIT_RPM, default 60/min)

- [x] **FEAT-008** Health monitoring & circuit breaker — DONE 2026-04-01
  - CircuitBreaker class for TTS (3 fails/30s) and prerender (2 fails/60s)
  - LiveKit connect retry with exponential backoff

- [x] **FEAT-009** Consistent API error format — DONE 2026-04-01
  - Custom exception handler returns `{"detail": "...", "code": "ERR_xxx"}`

## P3 — Nice to Have (3)

- [x] **FEAT-010** Prometheus metrics & observability — DONE 2026-04-01
  - prometheus-client added, /metrics endpoint, request count/latency/gauge instrumentation

- [x] **FEAT-011** API documentation — DONE 2026-04-01
  - Response models (HealthResponse, RegisterResponse, etc.), /docs and /redoc enabled

- [x] **FEAT-012** SQLite persistent state backend — DONE 2026-04-01
  - avatar_db.py with AvatarDB class, stores metadata + session logs in SQLite

---

## Progress

| Priority | Total | Done | Remaining |
|----------|-------|------|-----------|
| P0 | 2 | 2 | 0 |
| P1 | 4 | 4 | 0 |
| P2 | 3 | 3 | 0 |
| P3 | 3 | 3 | 0 |
| **Total** | **12** | **12** | **0** |
