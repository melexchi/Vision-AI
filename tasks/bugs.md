# Bug Tracker

All bugs found during full codebase scan (2026-04-01). Mark as FIXED with date and commit when resolved. Delete this file when all items are resolved.

## CRITICAL (8)

- [x] **BUG-001** Race condition on `avatar_cache` — FIXED 2026-04-01
  - Added `avatar_cache_lock = threading.Lock()`, wrapped all read/write accesses

- [x] **BUG-002** Race condition on `active_sessions` — FIXED 2026-04-01
  - Added `active_sessions_lock = threading.Lock()`, wrapped session creation and pop

- [x] **BUG-003** Race condition on `prerender_jobs` — FIXED 2026-04-01
  - Added `prerender_jobs_lock = threading.Lock()`, wrapped all writes in _run_prerender, _start_prerender, _ensure_skyreels_setup, delete_avatar

- [x] **BUG-004** Non-atomic pickle write — FIXED 2026-04-01
  - `_save_cache()` now writes to `.pkl.tmp` then `tmp_path.replace(final_path)` atomically

- [ ] **BUG-005** Unsafe pickle deserialization — `pickle.load()` on cache files allows arbitrary code execution
  - File: `ditto/ditto_api.py`
  - Impact: Remote code execution if attacker writes malicious .pkl
  - Fix: Replace pickle with safetensors/JSON or validate file integrity (deferred — requires format migration)

- [x] **BUG-006** Command injection via `os.system()` — FIXED 2026-04-01
  - Replaced `os.system(f-string)` with `subprocess.run(list_args)` in ditto_api.py and inference.py

- [x] **BUG-007** Unbounded `_clip_frame_cache` — FIXED 2026-04-01
  - Added LRU eviction with `_clip_cache_put()`/`_clip_cache_get()`, max 20 clips (configurable via `CLIP_CACHE_MAX` env var)

- [x] **BUG-008** Thread deadlock in pipeline `close()` — FIXED 2026-04-01
  - Set `stop_event` before joining, added `thread.join(timeout=10)` with warning on timeout

## HIGH (11)

- [x] **BUG-009** Path traversal — FIXED 2026-04-01
  - Added `_validate_file_path()` helper checking against allowed root directories

- [x] **BUG-010** No base64 size limits — FIXED 2026-04-01
  - Added 50MB size check before `b64decode()` in ditto, chatterbox, and smolvlm

- [x] **BUG-011** CORS misconfiguration — FIXED 2026-04-01
  - Set `allow_credentials=False` in chatterbox CORS middleware

- [x] **BUG-012** Bare `except: pass` — FIXED 2026-04-01
  - Replaced with `except Exception:` (no longer catches KeyboardInterrupt/SystemExit)

- [x] **BUG-013** `model_lock` unused — FIXED 2026-04-01
  - Replaced asyncio.Lock with threading.Lock, added double-check locking to getters

- [ ] **BUG-014** Blocking `load_ditto()` in async context — freezes event loop 20+ seconds on first request
  - File: `ditto/ditto_api.py` (lines 389-404)
  - Fix: Wrap in `run_in_executor()`, use async lock

- [ ] **BUG-015** No graceful shutdown — orphaned ffmpeg/SkyReels processes on crash, no SIGTERM handling
  - File: `ditto/ditto_api.py`
  - Fix: Add signal handlers, drain requests, cleanup subprocesses

- [ ] **BUG-016** No LiveKit reconnection logic — network hiccup kills session permanently
  - File: `ditto/ditto_api.py` (line 1719)
  - Fix: Add exponential backoff reconnection, session resume tokens

- [x] **BUG-017** SmolVLM hardcoded to cuda:0 — FIXED 2026-04-01
  - Device and model configurable via `SMOLVLM_DEVICE` and `SMOLVLM_MODEL` env vars

- [x] **BUG-018** Information disclosure — FIXED 2026-04-01
  - All `str(e)` in error responses replaced with generic messages; full errors logged server-side

- [ ] **BUG-019** No HTTP connection pooling — new `httpx.Client()` per TTS request
  - File: `ditto/ditto_api.py` (line 754)
  - Fix: Use global `httpx.AsyncClient` with connection pooling

## MEDIUM (19)

- [x] **BUG-020** Off-by-one in fade_out — FIXED 2026-04-01
  - Removed erroneous `-1` from `fade_start`, added `max(..., 1)` guard on division

- [ ] **BUG-021** Error response format inconsistent across services
  - Files: `ditto_api.py` (HTTPException), `smolvlm_server.py` (JSONResponse)
  - Fix: Standardize error envelope: `{"detail": "..."}` everywhere

- [ ] **BUG-022** No request rate limiting on any endpoint
  - Files: All servers
  - Fix: Add `slowapi` or custom token bucket middleware

- [ ] **BUG-023** No WebSocket message size limits or idle timeouts
  - Files: `ditto/ditto_api.py`, `chatterbox/api_server.py`
  - Fix: Add max message size, idle connection timeout (300s)

- [ ] **BUG-024** No pagination on `/avatars` endpoint
  - File: `ditto/ditto_api.py` (lines 855-870)
  - Fix: Add `?page=1&limit=50` query params

- [ ] **BUG-025** No config validation at startup
  - File: `ditto/ditto_api.py`
  - Fix: Pydantic settings model, validate paths/URLs at boot, fail fast

- [ ] **BUG-026** Audio resampling logic duplicated 3+ times
  - File: `ditto/ditto_api.py` (lines 1480-1488, 1036-1042, 1604)
  - Fix: Extract to `audio_utils.py`

- [ ] **BUG-027** Session memory leak — orphaned sessions on network drops, no TTL
  - File: `ditto/ditto_api.py` (lines 1762-1785)
  - Fix: Add session TTL with auto-cleanup (30min inactivity)

- [ ] **BUG-028** Magic numbers scattered without documentation
  - File: `ditto/ditto_api.py` (lines 73-84, 90-92, 102-115)
  - Fix: Move to config constants with docstrings

- [ ] **BUG-029** No API versioning on ditto and chatterbox
  - Files: `ditto/ditto_api.py`, `chatterbox/api_server.py`
  - Fix: Add `/v1/` prefix to all endpoints

- [x] **BUG-030** SmolVLM SSRF risk — FIXED 2026-04-01
  - Blocks localhost, 127.0.0.1, 10.x, 172.x, 192.168.x, and .internal hostnames

- [ ] **BUG-031** `print()` used everywhere instead of `logging` module
  - Files: All servers
  - Fix: Replace with `logging` module, add JSON formatter

- [ ] **BUG-032** No Prometheus metrics or latency tracking
  - Files: All servers
  - Fix: Add `prometheus-client`, instrument endpoints

- [ ] **BUG-033** No distributed tracing across service calls
  - Files: All servers
  - Fix: Add OpenTelemetry with request ID propagation

- [ ] **BUG-034** No error event reporting (Sentry/DataDog)
  - Files: All servers
  - Fix: Integrate error tracking service

- [x] **BUG-035** Unpinned dependencies — FIXED 2026-04-01
  - Added version ranges to all deps in chatterbox and smolvlm requirements.txt

- [ ] **BUG-036** No backpressure on audio stream — 1GB segment crashes server
  - File: `ditto/ditto_api.py` (lines 1560-1570)
  - Fix: Add max segment size limit, duration cap

- [ ] **BUG-037** Subprocess error output truncated to 300 chars
  - File: `ditto/ditto_api.py` (lines 534, 582)
  - Fix: Store full stderr, truncate only for display

- [x] **BUG-038** Missing `.env.example` — FIXED 2026-04-01
  - Created `.env.example` with all env vars, grouped by service, with descriptions

## LOW (5)

- [x] **BUG-039** Unused `tts_model` variable — FIXED 2026-04-01
  - Removed during model_lock refactor (BUG-013)

- [x] **BUG-040** Deprecated `streaming_sdk = None` — FIXED 2026-04-01
  - Removed entirely

- [x] **BUG-041** Hardcoded ports — FIXED 2026-04-01
  - Ditto: `DITTO_PORT` env var, SmolVLM: `SMOLVLM_PORT` env var

- [x] **BUG-042** No FPS validation — FIXED 2026-04-01
  - Added `Field(ge=1, le=60)` on fps and `Field(ge=1, le=50)` on sampling_timesteps

- [x] **BUG-043** Debug endpoints exposed — FIXED 2026-04-01
  - Gated behind `DITTO_DEBUG_ENDPOINTS=0` env var (enabled by default for dev)

---

## Progress

| Severity | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical | 8 | 7 | 1 (BUG-005 pickle deser — deferred, needs format migration) |
| High | 11 | 6 | 5 (BUG-014, 015, 016, 019 — need larger refactors) |
| Medium | 19 | 4 | 15 (BUG-022-029, 031-034, 036-037 — infra/observability work) |
| Low | 5 | 5 | 0 |
| **Total** | **43** | **22** | **21** |
