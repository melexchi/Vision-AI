# Feature Tracker

Recommended features from full codebase scan (2026-04-01). Mark as DONE with date and commit when implemented. Delete this file when all items are resolved.

## P0 — Must Have Before Production (2)

- [ ] **FEAT-001** Thread-safe state management
  - Add `threading.Lock()` to `avatar_cache`, `active_sessions`, `prerender_jobs`
  - Wrap all dict accesses in lock context managers
  - Addresses: BUG-001, BUG-002, BUG-003
  - Files: `ditto/ditto_api.py`

- [ ] **FEAT-002** Input validation layer
  - Base64 size limits (50MB max) on all decode endpoints
  - Path traversal protection with `Path.resolve()` whitelist
  - FPS/duration/sampling_timesteps bounds via Pydantic `Field()` validators
  - Avatar ID format validation (alphanumeric + hyphens only)
  - Audio segment max duration cap
  - Addresses: BUG-009, BUG-010, BUG-036, BUG-042
  - Files: `ditto/ditto_api.py`, `chatterbox/api_server.py`, `smolvlm/smolvlm_server.py`

## P1 — High Priority (4)

- [ ] **FEAT-003** Structured logging
  - Replace all `print()` with Python `logging` module
  - JSON log formatter for machine parsing
  - Request ID generation and propagation across service calls
  - Log levels: DEBUG for inference details, INFO for requests, WARNING for retries, ERROR for failures
  - Addresses: BUG-031
  - Files: All API servers

- [ ] **FEAT-004** Memory management with LRU eviction
  - LRU cache for `avatar_cache` with configurable max size (env: `MAX_CACHED_AVATARS`, default 50)
  - LRU cache for `_clip_frame_cache` with max 20 decoded clips
  - Session TTL with auto-cleanup after 30min inactivity
  - Periodic cleanup of completed/failed `prerender_jobs` older than 1 hour
  - Addresses: BUG-007, BUG-027
  - Files: `ditto/ditto_api.py`

- [ ] **FEAT-005** Graceful shutdown
  - SIGTERM/SIGINT signal handlers
  - Drain active HTTP requests (30s grace period)
  - Cancel and await all async tasks (publisher, audio receiver)
  - Kill orphaned subprocesses (ffmpeg, SkyReels prerender)
  - Close LiveKit rooms cleanly
  - Flush logs
  - Addresses: BUG-015
  - Files: `ditto/ditto_api.py`

- [ ] **FEAT-006** Configuration management
  - Create `.env.example` with all 12+ env vars documented
  - Pydantic `BaseSettings` model for config validation at startup
  - Fail fast with clear error messages for invalid/missing paths
  - Environment-specific defaults (dev vs production)
  - Addresses: BUG-025, BUG-038
  - Files: New `config.py` per service, `.env.example`

## P2 — Medium Priority (3)

- [ ] **FEAT-007** Rate limiting
  - Token bucket per client IP using `slowapi` or custom middleware
  - Limits: `/register` 5/min, `/generate` 10/min, `/start_session` 3/min
  - WebSocket connection limit: 50 concurrent per IP
  - Return `429 Too Many Requests` with `Retry-After` header
  - Addresses: BUG-022, BUG-023
  - Files: All API servers

- [ ] **FEAT-008** Health monitoring & circuit breaker
  - `/health` endpoints return model load status, memory usage, active sessions
  - Circuit breaker for Ditto → Chatterbox TTS calls (3 failures = open circuit, 30s cooldown)
  - Circuit breaker for Ditto → SkyReels prerender subprocess
  - Periodic model inference test (ping with tiny audio every 5min)
  - LiveKit reconnection with exponential backoff
  - Addresses: BUG-016
  - Files: `ditto/ditto_api.py`

- [ ] **FEAT-009** Consistent API error format
  - Standardize all error responses to `{"detail": "human message", "code": "ERROR_CODE"}`
  - Generic error messages to clients, full details to server logs
  - API versioning: add `/v1/` prefix to ditto and chatterbox endpoints
  - Addresses: BUG-018, BUG-021, BUG-029
  - Files: All API servers

## P3 — Nice to Have (3)

- [ ] **FEAT-010** Prometheus metrics & observability
  - Request latency histograms per endpoint (P50/P95/P99)
  - Error rate counters per endpoint and status code
  - Model inference time breakdown (wav2feat, audio2motion, decode, total)
  - Queue depth gauges for pipeline workers
  - Active session count, avatar cache size, memory usage
  - Addresses: BUG-032, BUG-033, BUG-034
  - Files: All API servers, new `metrics.py`

- [ ] **FEAT-011** API documentation
  - Auto-generated OpenAPI/Swagger from FastAPI (already built-in, just needs cleanup)
  - Add response models to all endpoints for accurate spec
  - Add example request/response bodies
  - Generate Postman collection from OpenAPI spec
  - Files: All API servers

- [ ] **FEAT-012** Database backend for persistent state
  - Replace in-memory `avatar_cache` with SQLite (or PostgreSQL for multi-instance)
  - Store avatar metadata, session history, inference logs
  - Enable horizontal scaling (multiple Ditto instances sharing state)
  - Migration path: pickle import tool for existing caches
  - Files: New `database.py`, `models.py`, migration scripts

---

## Progress

| Priority | Total | Done | Remaining |
|----------|-------|------|-----------|
| P0 | 2 | 0 | 2 |
| P1 | 4 | 0 | 4 |
| P2 | 3 | 0 | 3 |
| P3 | 3 | 0 | 3 |
| **Total** | **12** | **0** | **12** |
