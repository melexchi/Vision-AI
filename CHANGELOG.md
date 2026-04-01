# Changelog

All notable changes to this project are documented here.
Format: [DATE] [AUTHOR] Description

## [Unreleased]

### Added

- Initial Claude Code documentation layer (CLAUDE.md, docs/, tasks/, .claude/)
- `docs/CODEBASE_AUDIT.md` — Full codebase audit covering all 11 investigation areas
- `docs/ARCHITECTURE.md` — 12-section architecture document with diagrams and endpoint specs
- `docs/PATTERNS.md` — Code style guide with real snippets from every pattern in the codebase
- `docs/DEPLOY_LOG.md` — Deployment tracking template
- `docs/SSH_CONFIG.md` — SSH setup instructions for server access
- `docs/ACCESS.md` — Access and credentials guide for new team members
- 11 subdirectory `CLAUDE.md` files across ditto/, skyreels/, chatterbox/, smolvlm/
- Root `CLAUDE.md` with project rules, conventions, and safety protocols
- `.claudeignore` to protect context window from binary/generated files
- `tasks/todo.md` and `tasks/lessons.md` for task and knowledge tracking
- `.env.example` with all environment variables documented
- `CHANGELOG.md` (this file)

### Fixed

- **BUG-001/002/003** Thread-safe global state: added `threading.Lock()` to `avatar_cache`, `active_sessions`, `prerender_jobs`
- **BUG-004** Atomic pickle writes: `_save_cache()` now writes to `.tmp` then `replace()`
- **BUG-006** Command injection: replaced `os.system(f-string)` with `subprocess.run(list_args)` in ditto_api.py and inference.py
- **BUG-007** Memory leak: added LRU eviction to `_clip_frame_cache` (max 20, configurable via `CLIP_CACHE_MAX`)
- **BUG-008** Thread deadlock: pipeline `close()` now sets `stop_event` before joining with 10s timeout
- **BUG-009** Path traversal: added `_validate_file_path()` for user-supplied image/audio paths
- **BUG-010** DoS via large uploads: added 50MB base64 size limits on all endpoints
- **BUG-011** CORS misconfiguration: removed `allow_credentials=True` from chatterbox
- **BUG-012** Bare except: replaced with `except Exception:` in chatterbox WebSocket
- **BUG-013** Model lock race: replaced unused asyncio.Lock with threading.Lock + double-check locking
- **BUG-017** Hardcoded GPU: SmolVLM device configurable via `SMOLVLM_DEVICE` env var
- **BUG-018** Info disclosure: all error responses now return generic messages, full errors logged server-side
- **BUG-020** Off-by-one: fixed fade_out frame calculation in stream_pipeline_offline.py
- **BUG-030** SSRF: SmolVLM blocks private/internal URLs for image_url parameter
- **BUG-035** Unpinned deps: added version ranges to chatterbox and smolvlm requirements.txt
- **BUG-039/040** Dead code: removed unused `tts_model` and deprecated `streaming_sdk`
- **BUG-041** Hardcoded ports: ditto and smolvlm ports configurable via env vars
- **BUG-042** Input validation: added Pydantic `Field(ge=, le=)` bounds on fps and sampling_timesteps
- **BUG-043** Debug exposure: test endpoints gated behind `DITTO_DEBUG_ENDPOINTS` env var
- **BUG-005** Pickle safety: HMAC-SHA256 integrity verification on cache files with legacy auto-migration
- **BUG-014** Thread-safe load_ditto() with double-check locking
- **BUG-015** Graceful shutdown handler: stops all sessions, shuts down executor on SIGTERM
- **BUG-016** LiveKit reconnection: 3 retries with exponential backoff on room.connect()
- **BUG-019** HTTP connection pooling: reusable httpx.Client for TTS requests
- **BUG-022** Rate limiting: per-IP token bucket middleware (configurable RATE_LIMIT_RPM)
- **BUG-023** WebSocket security: 5min idle timeout + 100MB max message size
- **BUG-024** Pagination: `/avatars?limit=50&offset=0` with total count
- **BUG-025** Config validation: `_validate_config()` checks paths/URLs at startup
- **BUG-026** Audio dedup: extracted `_to_mono_int16()` and `_resample_int16()` to module level
- **BUG-027** Session TTL: background cleanup loop with configurable SESSION_TTL_SECONDS (30min default)
- **BUG-028** Constants documented: inline docstrings on all pipeline magic numbers
- **BUG-030** SSRF: SmolVLM blocks private/internal URLs
- **BUG-031** Structured logging: replaced all print() with Python logging module in ditto and chatterbox
- **BUG-036** Audio backpressure: 50MB max segment size limit on LiveKit DataStream
- **BUG-037** Subprocess errors: full stderr in logs, 1000-char storage (was 300)

### Added (Features)

- **FEAT-004** Avatar cache LRU eviction (`MAX_CACHED_AVATARS`, default 50) + prerender job cleanup (1hr expiry)
- **FEAT-008** CircuitBreaker class for TTS and prerender calls (auto-open after N failures, cooldown reset)
- **FEAT-009** Consistent error format: all errors return `{"detail": "...", "code": "ERR_xxx"}`
- **FEAT-010** Prometheus metrics: `/metrics` endpoint with request count, latency histograms, active session gauge
- **FEAT-011** OpenAPI response models (HealthResponse, RegisterResponse, AvatarListResponse, SessionResponse, ErrorResponse)
- **FEAT-012** SQLite metadata store (`ditto/avatar_db.py`) for persistent avatar + session tracking
- 23 automated tests: SQLite CRUD, thread safety, input validation, circuit breaker, LRU cache, SSRF protection

### Removed

- `streaming_sdk` deprecated global variable
- `tts_model` unused global variable
