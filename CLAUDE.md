# Project Overview

Vision-AI is a self-hosted GPU inference stack for real-time AI talking avatars. Four microservices — Ditto (lip-sync streaming), Chatterbox (TTS), SmolVLM (camera vision), and SkyReels (expression clips) — combine to stream 25fps avatar video over LiveKit WebRTC with ~200ms latency. All inference runs on NVIDIA GPUs (A100 recommended) with TensorRT optimization. Deployed on RunPod GPU cloud.

# Tech Stack

- **Web framework:** FastAPI + Uvicorn (all services)
- **ML frameworks:** PyTorch 2.8, TensorRT 10.7, ONNX Runtime, HuggingFace Diffusers/Transformers
- **Real-time:** LiveKit SDK (WebRTC video/audio streaming)
- **Package management:** uv (Ditto), pip (others), Conda (legacy)
- **Containerization:** Docker (nvidia/cuda:12.8.0 base)
- **Database:** None — in-memory dicts + pickle disk caches
- **Auth:** None — assumes private network or external reverse proxy

# Architecture

4 services: Ditto (:8181) ↔ Chatterbox (:8080), Ditto → SkyReels (subprocess), SmolVLM (:8282) standalone.
Ditto is the hub — receives agent audio via LiveKit, generates lip-synced video, streams to browsers.
See `docs/ARCHITECTURE.md` for full diagrams, data flows, and endpoint specs.

# Directory Structure

- `ditto/` — Real-time lip-sync avatar streaming (Port 8181), main service
- `chatterbox/` — Neural TTS with voice cloning (Port 8080)
- `smolvlm/` — Camera vision VLM service (Port 8282)
- `skyreels/` — Expression clip generation via 8B diffusion model
- `docs/` — Architecture, audit, patterns, deployment docs
- `tasks/` — Task tracking and lessons learned

# Key Commands

```bash
# Ditto (primary service)
cd ditto && bash setup.sh                                        # One-time setup
cd ditto && uv run uvicorn ditto_api:app --host 0.0.0.0 --port 8181  # Run server
cd ditto && docker build -t ditto .                              # Build container

# Chatterbox
cd chatterbox && pip install -r requirements.txt
cd chatterbox && uvicorn api_server:app --host 0.0.0.0 --port 8080

# SmolVLM
cd smolvlm && pip install -r requirements.txt
cd smolvlm && uvicorn smolvlm_server:app --host 0.0.0.0 --port 8282

# SkyReels
cd skyreels && bash setup.sh                                     # One-time setup (~15GB)
cd skyreels && .venv/bin/python app.py                           # Gradio UI
```

# Coding Conventions

- **Files:** `snake_case.py`, directories `snake_case/` or `kebab-case/`
- **Classes:** `PascalCase` (e.g., `RegisterAvatarRequest`, `Wav2Feat`)
- **Functions:** `snake_case`, private `_prefixed` (e.g., `_save_cache`)
- **Constants:** `ALL_CAPS` (e.g., `CACHE_DIR`, `DEFAULT_FPS`)
- **Config:** `os.environ.get("NAME", default)` at module level with `Path()` wrapping
- **Imports:** stdlib → sys.path → third-party → try/except optional → relative internal
- **API routes:** async FastAPI handlers, Pydantic BaseModel for request bodies
- **Errors:** `HTTPException(status_code=400/500, detail=str)`

# Patterns

See `docs/PATTERNS.md` for annotated code examples with file paths and line numbers covering: API routes, data access, error handling, middleware, config, module organization, background jobs, response formats, naming, and imports.

# Testing

No automated test framework is currently configured. Manual testing only:
- `skyreels/skyreels_a1/src/lmk3d_test.py` — CLI face animation test
- `ditto/test_viewer.html` — Browser WebSocket debug viewer
- `WS /ws/test_frames` — Ditto test endpoint (sine-wave → frames)

# Task Management

- Before starting work, write plan to `tasks/todo.md`
- Track progress by marking items complete
- After ANY correction or mistake, update `tasks/lessons.md` with a rule that prevents it
- After completing work, add entry to `CHANGELOG.md`

# Git Workflow

- Always create a feature branch: `feature/[your-name]/[short-description]`
- Never commit directly to main
- Every PR must have a clear description of what changed and why
- Run tests before pushing
- Request review from at least one team member

# Important Rules

- NEVER modify code without an approved plan
- NEVER skip tests
- ALWAYS check `docs/PATTERNS.md` before creating new patterns
- ALWAYS update `CHANGELOG.md` with your changes
- Do not touch: `.claude/skills/`, `node_modules/`, `.git/`, `dist/`, `build/`, `.env` files, `checkpoints/`, `pretrained_models/`

# Frontend Design Rules (Impeccable)

- For ANY frontend/UI work, run /audit after /review and /polish before final commit
- Never use Inter, Arial, Roboto, or system fonts as primary typeface — pick distinctive fonts
- Never use pure gray — always tint neutrals toward the brand color
- Never nest cards inside cards
- Never use gray text on colored backgrounds — check contrast
- Never use purple gradients as default — commit to a project-specific color palette
- Never use bounce/elastic easing — it feels dated
- See `.claude/skills/frontend-design/` for full design reference

# Bulk Operation Safety

- NEVER run bulk find-and-replace (sed, grep -rl | xargs) without excluding: `.claude/skills/`, `node_modules/`, `.git/`, `dist/`, `build/`, `checkpoints/`, `pretrained_models/`
- Safe bulk rename: `grep -rl 'old' --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=dist --exclude-dir=build --exclude-dir=.claude/skills --exclude-dir=checkpoints . | xargs sed -i 's/old/new/g'`
- ALWAYS show the list of files that will be affected BEFORE running any bulk operation
- ALWAYS ask for confirmation before executing bulk changes

# Security Rules (Enforced on Every Task)

- NEVER store auth tokens in localStorage — use httpOnly cookies only
- NEVER return stack traces, file paths, or SQL errors in API responses
- NEVER build SQL queries with string concatenation — always use parameterized queries
- NEVER commit .env files or hardcode secrets in source code
- NEVER serve user-uploaded files without MIME type and size validation
- ALWAYS validate environment variables at app startup
- ALWAYS add server-side validation — client-side validation is not security
- ALWAYS use HTTPS — no mixed content allowed
- ALWAYS rate-limit API endpoints (especially auth, payment, and public routes)

# Deployment Rules

- Server: Claude can deploy and test autonomously. For destructive DB ops (DROP, TRUNCATE, DELETE without WHERE), show command and wait for APPROVED.
- ALWAYS backup database before running migrations — no exceptions
- NEVER auto-fix without circuit breaker: max 3 auto-fix cycles, then STOP and report
- If circuit breaker fires, run /rollback — NEVER leave a broken deployment live
- Log every deployment to `docs/DEPLOY_LOG.md`
- Before ANY operation touching 5+ files, show the file list and wait for APPROVED

# Testing Rules

- Every new API endpoint MUST have at least one automated test
- Tests must verify behavior (what SHOULD happen), not just implementation
- After human QA finds a bug, add a regression test

# Subdirectory Docs

Each service and engine subdir has its own CLAUDE.md: `ditto/`, `ditto/ditto-talkinghead/`, `ditto/ditto-talkinghead/core/{atomic_components,aux_models,models,utils}/`, `skyreels/`, `skyreels/{skyreels_a1,diffposetalk}/`, `chatterbox/`, `smolvlm/`.

# Context Window Budget

- Root CLAUDE.md: under 150 lines. Each subdirectory CLAUDE.md: under 80 lines.
- `tasks/lessons.md`: prune entries older than 30 days to an archive file
- `.claudeignore`: keeps build artifacts, dependencies, and large data out of context
- One task per session — start fresh, don't carry over context from previous tasks
