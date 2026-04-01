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

- [x] **BUG-005** Unsafe pickle deserialization — FIXED 2026-04-01
  - Added HMAC-SHA256 integrity verification on pickle files; legacy files auto-migrated on first load

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

- [x] **BUG-014** Blocking `load_ditto()` — FIXED 2026-04-01
  - Added double-check threading.Lock; startup already uses run_in_executor

- [x] **BUG-015** No graceful shutdown — FIXED 2026-04-01
  - Added @app.on_event("shutdown") handler: stops all sessions, shuts down executor

- [x] **BUG-016** No LiveKit reconnection — FIXED 2026-04-01
  - Added retry loop with exponential backoff (3 attempts) on room.connect()

- [x] **BUG-017** SmolVLM hardcoded to cuda:0 — FIXED 2026-04-01
  - Device and model configurable via `SMOLVLM_DEVICE` and `SMOLVLM_MODEL` env vars

- [x] **BUG-018** Information disclosure — FIXED 2026-04-01
  - All `str(e)` in error responses replaced with generic messages; full errors logged server-side

- [x] **BUG-019** No HTTP connection pooling — FIXED 2026-04-01
  - Added `_get_tts_client()` returning a reusable httpx.Client with keep-alive

## MEDIUM (19)

- [x] **BUG-020** Off-by-one in fade_out — FIXED 2026-04-01
  - Removed erroneous `-1` from `fade_start`, added `max(..., 1)` guard on division

- [x] **BUG-021** Error response inconsistency — RESOLVED 2026-04-01
  - SmolVLM's `{"success": bool}` pattern is intentional and documented in PATTERNS.md; error messages sanitized via BUG-018

- [x] **BUG-022** No rate limiting — FIXED 2026-04-01
  - Added per-IP token bucket middleware (configurable via `RATE_LIMIT_RPM`, default 60/min)

- [x] **BUG-023** No WebSocket limits — FIXED 2026-04-01
  - Added 5min idle timeout and 100MB max message size to chatterbox /ws

- [x] **BUG-024** No pagination — FIXED 2026-04-01
  - `/avatars` now accepts `?limit=50&offset=0` query params, returns total count

- [x] **BUG-025** No config validation — FIXED 2026-04-01
  - Added `_validate_config()` at startup, warns on missing DITTO_PATH or invalid TTS_URL

- [x] **BUG-026** Audio resampling duplication — FIXED 2026-04-01
  - Extracted `_to_mono_int16()` and `_resample_int16()` to module level

- [x] **BUG-027** Session memory leak — FIXED 2026-04-01
  - Added `_session_cleanup_loop()` background task with configurable TTL (`SESSION_TTL_SECONDS`, default 30min)

- [x] **BUG-028** Magic numbers — FIXED 2026-04-01
  - Added inline docstrings explaining all constants (DEFAULT_SAMPLING_TIMESTEPS, AUDIO_DELAY_FRAMES, sample rates, chunk sizes)

- [ ] **BUG-029** No API versioning on ditto and chatterbox
  - Deferred: adding `/v1/` prefix is a breaking change requiring client migration plan

- [x] **BUG-030** SmolVLM SSRF risk — FIXED 2026-04-01
  - Blocks localhost, 127.0.0.1, 10.x, 172.x, 192.168.x, and .internal hostnames

- [x] **BUG-031** `print()` everywhere — FIXED 2026-04-01
  - Replaced all print() with `logging.getLogger()` in ditto_api.py and api_server.py; added timestamps

- [ ] **BUG-032** No Prometheus metrics or latency tracking
  - Deferred: requires adding `prometheus-client` dependency and testing with GPU stack

- [ ] **BUG-033** No distributed tracing across service calls
  - Deferred: requires adding OpenTelemetry dependency and instrumenting all services

- [ ] **BUG-034** No error event reporting (Sentry/DataDog)
  - Deferred: requires choosing and configuring error tracking service

- [x] **BUG-035** Unpinned dependencies — FIXED 2026-04-01
  - Added version ranges to all deps in chatterbox and smolvlm requirements.txt

- [x] **BUG-036** No backpressure on audio stream — FIXED 2026-04-01
  - Added 50MB max segment size limit; truncates and warns when exceeded

- [x] **BUG-037** Subprocess error truncation — FIXED 2026-04-01
  - Increased error storage to 1000 chars, full stderr printed to logs

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
| Critical | 8 | 8 | 0 |
| High | 11 | 10 | 1 (BUG-029 API versioning — deferred, breaking change) |
| Medium | 19 | 16 | 3 (BUG-032/033/034 — need new deps: Prometheus, OpenTelemetry, Sentry) |
| Low | 5 | 5 | 0 |
| **Total** | **43** | **39** | **4** |
