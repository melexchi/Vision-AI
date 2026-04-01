# Ditto — Real-Time Lip-Sync Avatar Streaming

## Purpose

Ditto is the primary service in the Vision-AI stack. It registers avatar portraits, extracts facial features, and streams real-time lip-synced video at 25fps over LiveKit WebRTC. It also serves pre-rendered expression clips and supports offline video generation from audio or text.

## Key Files

- `ditto_api.py` — FastAPI server with all 15 REST + 2 WebSocket endpoints, LiveKit session management, avatar caching, and clip serving (1956 lines, the largest file in the project)
- `prerender_clips.py` — Subprocess script that calls SkyReels-A1 to generate idle/thinking/greeting clips for registered avatars
- `Dockerfile` — Production container based on `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` for RunPod deployment
- `setup.sh` — Idempotent setup: installs uv, syncs deps, downloads weights, builds TRT engines and Cython extensions
- `download_weights.sh` — Downloads model weights from HuggingFace `digital-avatar/ditto-talkinghead` with parallel transfer
- `pyproject.toml` — Dependencies with custom PyTorch/NVIDIA pip indices
- `uv.lock` — Reproducible lock file (2278 lines)
- `ditto-talkinghead/` — Core inference engine (see its own CLAUDE.md)

## Data Flow

```
Client/Agent → POST /register (image) → Feature extraction → avatar_cache + disk pickle
Client/Agent → POST /generate (audio) → wav2feat → audio2motion → stitch → warp → decode → MP4
Agent → LiveKit ByteStream (audio) → Ditto pipeline → LiveKit WebRTC (video+audio) → Browser
```

## Dependencies

- **Depends on:** `ditto-talkinghead/` (inference engine), Chatterbox (HTTP for `/generate_from_text`), SkyReels (subprocess for clip prerendering)
- **Depended on by:** Client browsers (LiveKit WebRTC), AI agents (LiveKit DataStream + RPC)

## Conventions

- All state is in module-level dicts (`avatar_cache`, `active_sessions`, `prerender_jobs`) — no database
- Environment config via `os.environ.get("NAME", default)` with `Path()` wrapping
- Background jobs use `ThreadPoolExecutor(max_workers=1)` for GPU serialization
- LiveKit topics: `lk.audio_stream`, `lk.clear_buffer`, `lk.playback_finished`

## Common Commands

```bash
bash setup.sh                    # One-time setup (deps + weights + TRT engines)
uv run uvicorn ditto_api:app --host 0.0.0.0 --port 8181   # Start server
bash download_weights.sh         # Re-download model weights
docker build -t ditto .          # Build Docker image
```
