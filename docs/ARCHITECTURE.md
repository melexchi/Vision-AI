# Vision-AI Architecture

**Last updated:** 2026-04-01

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Directory Map](#3-directory-map)
4. [Database Schema](#4-database-schema)
5. [API Surface](#5-api-surface)
6. [Authentication & Authorization](#6-authentication--authorization)
7. [Background Jobs / Queues](#7-background-jobs--queues)
8. [Third-Party Integrations](#8-third-party-integrations)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Key Environment Variables](#10-key-environment-variables)
11. [Real-Time / Event Flows](#11-real-time--event-flows)
12. [Server Access](#12-server-access)

---

## 1. System Overview

Vision-AI is a self-hosted GPU inference stack that powers real-time AI talking avatars. It combines four microservices — live lip-sync video streaming (Ditto), neural text-to-speech with voice cloning (Chatterbox), pre-rendered facial expression clip generation (SkyReels), and camera-frame understanding via a vision-language model (SmolVLM). The primary real-time path streams 25fps portrait video over LiveKit WebRTC with ~200ms latency, driven by incoming audio from a conversational AI agent. All inference runs on NVIDIA GPUs (A100 recommended) using TensorRT-optimized models, and the system is designed for deployment on RunPod GPU cloud.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            VISION-AI STACK                              │
│                                                                         │
│  ┌───────────────────┐       ┌────────────────────────────────────────┐ │
│  │   Chatterbox TTS  │       │           Ditto Avatar                │ │
│  │     Port 8080     │◄──────│           Port 8181                   │ │
│  │                   │ HTTP  │                                        │ │
│  │  FastAPI + WS     │ /tts  │  FastAPI + WS + LiveKit WebRTC        │ │
│  │  - Multilingual   │       │  - Avatar registration & caching      │ │
│  │  - Turbo          │       │  - Real-time lip-sync pipeline        │ │
│  │  - Voice cloning  │       │  - Offline video generation           │ │
│  └───────────────────┘       │  - Pre-rendered clip serving          │ │
│                              └──────────┬─────────────────────────────┘ │
│                                         │ subprocess                    │
│                                         ▼                               │
│  ┌───────────────────┐       ┌──────────────────────┐                  │
│  │   SmolVLM         │       │   SkyReels-A1        │                  │
│  │   Port 8282       │       │   (batch / Gradio)   │                  │
│  │                   │       │                      │                  │
│  │  FastAPI           │       │  8B-param diffusion  │                  │
│  │  - Image queries  │       │  - idle clips        │                  │
│  │  - 2.2B VLM       │       │  - expression clips  │                  │
│  └───────────────────┘       └──────────────────────┘                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════════ │
│                         EXTERNAL CONNECTIONS                            │
│                                                                         │
│  ┌───────────────────┐       ┌──────────────────────┐                  │
│  │  LiveKit Server   │◄─────►│  AI Agent            │                  │
│  │  (WebRTC SFU)     │       │  (sends audio via    │                  │
│  │                   │       │   DataStream, receives│                  │
│  │  wss://...        │       │   avatar video/audio) │                  │
│  └───────┬───────────┘       └──────────────────────┘                  │
│          │                                                              │
│          ▼                                                              │
│  ┌───────────────────┐                                                  │
│  │  Client Browser   │                                                  │
│  │  (WebRTC viewer)  │                                                  │
│  └───────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Inter-Service Communication

| From | To | Protocol | Purpose |
|---|---|---|---|
| Ditto | Chatterbox | HTTP (`TTS_URL`) | Text-to-speech for `/generate_from_text` endpoint |
| Ditto | SkyReels | Subprocess (`prerender_clips.py`) | Generate idle/thinking/greeting expression clips |
| AI Agent | Ditto | LiveKit ByteStream (`lk.audio_stream`) | Stream audio chunks for real-time lip-sync |
| Ditto | AI Agent | LiveKit RPC (`lk.playback_finished`) | Report playback position and interruption status |
| AI Agent | Ditto | LiveKit RPC (`lk.clear_buffer`) | Interrupt current speech playback |
| Client | Ditto | LiveKit WebRTC | Receive 25fps avatar video + audio stream |
| Client | SmolVLM | HTTP REST | Send camera frames for vision-language analysis |

---

## 3. Directory Map

### Top-Level

| Path | Purpose |
|---|---|
| `ditto/` | Real-time lip-sync avatar streaming service (Port 8181) |
| `skyreels/` | Expression clip generation via SkyReels-A1 diffusion model |
| `chatterbox/` | Neural TTS service with voice cloning (Port 8080) |
| `smolvlm/` | Vision-language model for camera frame understanding (Port 8282) |
| `docs/` | Project documentation (audit, architecture) |
| `.claude/` | Claude Code local configuration |
| `README.md` | Project overview, hardware requirements, architecture diagrams |
| `.gitignore` | Excludes model weights, Python cache, .env files |

### Ditto (Second Level)

| Path | Purpose |
|---|---|
| `ditto/ditto_api.py` | FastAPI server — all 15 REST + 1 WS endpoints, LiveKit streaming, session management (79K, 1956 lines) |
| `ditto/prerender_clips.py` | SkyReels-A1 clip generation script called as subprocess (17K) |
| `ditto/Dockerfile` | Docker container for RunPod GPU deployment |
| `ditto/setup.sh` | Idempotent setup: uv deps, model weights, TRT engines, Cython build |
| `ditto/download_weights.sh` | HuggingFace model weight downloader with parallel transfer |
| `ditto/pyproject.toml` | Python dependencies + custom PyTorch/NVIDIA pip indices |
| `ditto/uv.lock` | Reproducible dependency lock file (2278 lines) |
| `ditto/requirements.txt` | Pip dependencies (subset) |
| `ditto/test_viewer.html` | HTML WebSocket debug viewer |
| `ditto/ditto-talkinghead/` | Core inference engine (see below) |

### Ditto TalkingHead Engine (Third Level)

| Path | Purpose |
|---|---|
| `ditto-talkinghead/core/atomic_components/` | 12 modular pipeline stages: `audio2motion`, `avatar_registrar`, `wav2feat`, `motion_stitch`, `warp_f3d`, `decode_f3d`, `putback`, `source2info`, `condition_handler`, `writer`, `loader`, `cfg` |
| `ditto-talkinghead/core/aux_models/` | Face detection (BlazeFace, InsightFace), landmarks (MediaPipe 478, 106, 203), audio encoder (HuBERT streaming) |
| `ditto-talkinghead/core/models/` | Neural networks: LMDM diffusion, appearance/motion extractors, decoder, warp network, stitch network |
| `ditto-talkinghead/core/models/modules/` | Sub-modules: LMDM transformer, ConvNextV2, dense motion, SPADE generator, stitching network |
| `ditto-talkinghead/core/utils/` | TRT wrapper, face cropping, eye info, masks, model loader (ONNX/TRT/PyTorch) |
| `ditto-talkinghead/core/utils/blend/` | Cython/C-accelerated image blending (blend.pyx + blend_impl.c/h) |
| `ditto-talkinghead/scripts/cvt_onnx_to_trt.py` | ONNX → TensorRT engine converter with GridSample3D plugin support |
| `ditto-talkinghead/inference.py` | Standalone inference entry point |
| `ditto-talkinghead/stream_pipeline_online.py` | Online (real-time) pipeline runner |
| `ditto-talkinghead/stream_pipeline_offline.py` | Offline (batch) pipeline runner with multi-threaded worker queues |
| `ditto-talkinghead/environment.yaml` | Legacy Conda environment spec |

### SkyReels (Second Level)

| Path | Purpose |
|---|---|
| `skyreels/app.py` | Gradio web UI for interactive video generation |
| `skyreels/inference.py` | Image-to-video inference (video driving) |
| `skyreels/inference_audio.py` | Audio-to-video inference |
| `skyreels/inference_audio_long_video.py` | Long-form audio-to-video generation |
| `skyreels/inference_long_video.py` | Long-form video generation |
| `skyreels/setup.sh` | One-time setup: venv, PyTorch, pytorch3d, model downloads |
| `skyreels/requirements.txt` | Pip dependencies |
| `skyreels/skyreels_a1/` | Main SkyReels-A1 pipeline: I2V pipeline, 3D transformer (36K), DDIM solver, landmark preprocessing |
| `skyreels/skyreels_a1/models/transformer3d.py` | 3D transformer for video diffusion (8B params) |
| `skyreels/skyreels_a1/src/` | Frame interpolation (FILM), SMIRK encoder, FLAME 3D face model, MediaPipe integration |
| `skyreels/diffposetalk/` | Audio-to-FLAME motion: DiffPoseTalk model, HuBERT/Wav2Vec2 feature extractors |
| `skyreels/diffposetalk/utils/` | Rotation conversions, 3D mesh rendering, media utilities |
| `skyreels/eval/` | Evaluation metrics: ArcFace similarity, expression quality, pose accuracy |
| `skyreels/scripts/demo.py` | Pipeline initialization and inference helpers |
| `skyreels/assets/` | Example media: driving audio (6 WAV), driving video (8 MP4), reference portraits (20 PNG) |

### Chatterbox (Second Level)

| Path | Purpose |
|---|---|
| `chatterbox/api_server.py` | FastAPI server with REST + WebSocket TTS endpoints (304 lines) |
| `chatterbox/requirements.txt` | Dependencies: chatterbox-tts, torch, torchaudio, FastAPI |

### SmolVLM (Second Level)

| Path | Purpose |
|---|---|
| `smolvlm/smolvlm_server.py` | FastAPI server for image understanding queries (195 lines) |
| `smolvlm/pyproject.toml` | Project configuration |
| `smolvlm/requirements.txt` | Dependencies: transformers, torch, FastAPI, Pillow, httpx |

---

## 4. Database Schema

**This project uses no SQL database.** All state is managed through in-memory Python dictionaries and filesystem-based caches. There are no migrations, no ORM, and no database connections.

### Ditto State Stores

#### `avatar_cache` (In-Memory Dictionary)

The primary data structure. Keyed by `avatar_id` (string), valued as `source_info` dictionaries (~9MB each).

| Field | Type | Description |
|---|---|---|
| `avatar_id` | `str` | Unique identifier (UUID or user-provided) |
| `x_s_info_lst` | `list[dict]` | Per-frame motion parameters (keypoints, expressions) |
| `img_rgb_lst` | `list[np.ndarray]` | RGB frame arrays from source image/video |
| `is_image_flag` | `bool` | Whether source is a single image or video |
| `c_pos_s_lst` | `list` | Frame-specific position features |
| `c_con_s_lst` | `list` | Frame-specific conditioning features |
| `t_bbox_s_lst` | `list` | Frame-specific bounding boxes |
| `blink_lst` | `list` | Eye blink tracking data |
| `eye_open_n_lst` | `list` | Eye openness normalization data |
| Additional 10+ fields | Various | Appearance features, landmarks, crop parameters |

#### `active_sessions` (In-Memory Dictionary)

Keyed by `session_id` (UUID string).

| Field | Type | Description |
|---|---|---|
| `room` | `lk_rtc.Room` | LiveKit room connection |
| `video_source` | `lk_rtc.VideoSource` | Published video source |
| `audio_source` | `lk_rtc.AudioSource` | Published audio source (24kHz mono) |
| `av_sync` | `lk_rtc.AVSynchronizer` | Audio/video synchronization handler |
| `audio_24k_chunks` | `list[np.ndarray]` | Buffered audio from agent |
| `audio_24k_buffer` | `np.ndarray` | Concatenated audio buffer |
| `avatar_id` | `str` | Active avatar for this session |
| `fps` | `int` | Target frame rate (default 25) |
| `sampling_timesteps` | `int` | DDIM diffusion steps |
| `publisher_task` | `asyncio.Task` | Background frame publisher |
| `audio_recv_task` | `asyncio.Task` | Background audio receiver |
| `output_queue` | `asyncio.Queue` | Pipeline output frames |
| `_speech_interrupted` | `bool` | Interrupt flag from agent RPC |

#### `prerender_jobs` (In-Memory Dictionary)

Keyed by `avatar_id`.

| Field | Type | Description |
|---|---|---|
| `status` | `str` | `"pending"` / `"setting_up"` / `"running"` / `"done"` / `"failed"` |
| `started` | `float` | `time.time()` when job began |
| `finished` | `float` | `time.time()` when job completed |
| `error` | `str` | Error message if failed |

### Filesystem Persistence

| Path Pattern | Format | Contents | Lifetime |
|---|---|---|---|
| `{CACHE_DIR}/{avatar_id}.pkl` | Pickle | Serialized `source_info` dict | Persistent across restarts |
| `{IMAGES_DIR}/{avatar_id}.png` | PNG | Original source portrait image | Persistent |
| `{CLIPS_DIR}/{avatar_id}_{clip_type}_{variant}.mp4` | MP4 | Pre-rendered expression clips | Persistent |
| `{CLIPS_DIR}/{avatar_id}_{clip_type}_{variant}_decoded/` | Directory of PNGs | Decoded clip frames for fast playback | Persistent (lazy-created) |

**Clip types:** `idle` (6 variants), `thinking` (4), `lookingup` (4), `lippurse` (4), `greeting` (4)

### Chatterbox State

| Store | Type | Contents |
|---|---|---|
| `tts_model` | Global var | Lazy-loaded `ChatterboxMultilingualTTS` |
| `turbo_model` | Global var | Lazy-loaded `ChatterboxTurboTTS` |
| `{TEMP_DIR}/chatterbox_api/{session_id}/` | Disk | Ephemeral voice prompt WAVs (cleaned up on WS disconnect) |

### SmolVLM State

| Store | Type | Contents |
|---|---|---|
| `vlm_model` | Global var | Lazy-loaded `AutoModelForImageTextToText` (SmolVLM2-2.2B-Instruct) |
| `vlm_processor` | Global var | Lazy-loaded `AutoProcessor` |

### Relationships

```
avatar_cache[avatar_id]
    │
    ├── 1:1 ── {CACHE_DIR}/{avatar_id}.pkl        (disk backup)
    ├── 1:1 ── {IMAGES_DIR}/{avatar_id}.png        (source image)
    ├── 1:N ── {CLIPS_DIR}/{avatar_id}_*.mp4       (pre-rendered clips)
    └── 1:N ── active_sessions (multiple sessions can use same avatar)

active_sessions[session_id]
    │
    ├── N:1 ── avatar_cache[avatar_id]              (references one avatar)
    ├── 1:1 ── LiveKit Room                          (one room per session)
    └── 1:1 ── prerender_jobs[avatar_id]             (optional, tracks clip rendering)
```

---

## 5. API Surface

### Chatterbox — Text-to-Speech (Port 8080)

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| GET | `/health` | Model load status | — | `{status, multilingual_loaded, turbo_loaded}` |
| POST | `/api/tts` | Generate speech from text | Form: `text`, `language`, `voice_prompt` (file), `exaggeration`, `cfg_weight`, `model_type` | `audio/wav` binary |
| WS | `/ws` | Streaming TTS with voice cloning session | See [Real-Time Flows](#11-real-time--event-flows) | WebSocket messages |

### Ditto — Avatar Streaming (Port 8181)

**Avatar Management:**

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| POST | `/register` | Register avatar from portrait image | `{avatar_id?, image_base64?, image_path?, prerender_clips: bool}` | `{status, avatar_id, inference_ready, clips_ready, prerender_status, size_mb}` |
| GET | `/avatars` | List all registered avatars | — | `{avatars: [{avatar_id, frames, is_image, size_mb, on_disk, inference_ready, clips_ready, prerender_status}], total}` |
| GET | `/avatars/{avatar_id}/status` | Single avatar status | — | `{avatar_id, inference_ready, clips_ready, prerender}` |
| GET | `/avatars/{avatar_id}/resolution` | Output video dimensions | — | `{width, height}` |
| DELETE | `/avatars/{avatar_id}` | Evict avatar (memory + disk + clips) | — | `{status: "deleted", avatar_id}` |

**Video Generation:**

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| POST | `/generate` | Generate video from audio | `{avatar_id, audio_base64?, audio_path?, sampling_timesteps?, fps?}` | `video/mp4` binary |
| POST | `/generate_from_text` | Generate video from text (calls Chatterbox) | `{avatar_id, text, voice: "tara"}` | `video/mp4` binary |
| POST | `/generate_frames` | Stream raw JPEG frames from PCM audio | Headers: `X-Avatar-Id`, `X-Sample-Rate`; Body: raw int16 PCM | `application/octet-stream` (length-prefixed JPEGs) |

**Clip Management:**

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| GET | `/clips/{avatar_id}/{clip_type}` | Serve pre-rendered clip (random variant) | — | `video/mp4` (200), or 202 (rendering), 404 (not found) |
| POST | `/avatars/{avatar_id}/prerender` | Trigger clip pre-rendering | — | `{avatar_id, prerender_status: "submitted"}` |

**Real-time Streaming:**

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| POST | `/start_session` | Start LiveKit streaming session | `{avatar_id, livekit_url, livekit_token, fps?, sampling_timesteps?, agent_identity?}` | `{session_id, status: "connected", avatar_id, room}` |
| POST | `/stop_session/{session_id}` | Stop streaming session | — | `{session_id, status: "stopped"}` |

**Utility:**

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| GET | `/health` | Service health | — | `{status, model, cached_avatars, avatar_ids, sampling_timesteps, fps}` |
| GET | `/test_viewer` | HTML test page | — | `text/html` |
| WS | `/ws/test_frames` | Test WebSocket (sine-wave → frames) | Query: `avatar_id`, `duration` | WebSocket: base64 JPEG frames + WAV audio |

### SmolVLM — Camera Vision (Port 8282)

| Method | Path | Description | Request | Response |
|---|---|---|---|---|
| GET | `/health` | Service status | — | `{status: "ok", service: "smolvlm"}` |
| POST | `/v1/query` | Query with image + prompt | `{image_base64?, image_url?, prompt}` | `{success, response, prompt}` |
| POST | `/v1/query/upload` | Query with uploaded image | Form: `image` (file), `prompt` | `{success, response, prompt}` |
| POST | `/v1/unload` | Free GPU VRAM | — | `{status: "unloaded"}` |

### SkyReels — Clip Generation

No REST API. Available as:
- **Gradio UI** (`skyreels/app.py`) — interactive portrait animation
- **CLI scripts** (`inference.py`, `inference_audio.py`, etc.) — batch processing
- **Subprocess** — called by Ditto's `prerender_clips.py`

---

## 6. Authentication & Authorization

**There is no authentication or authorization implemented in any service.**

### What Exists

- **CORS**: All three HTTP services enable wildcard CORS — `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` (Ditto and Chatterbox explicitly; SmolVLM uses FastAPI defaults).
- **LiveKit JWT**: The only token-based mechanism. Clients pass a `livekit_token` (JWT signed by the LiveKit server) to `POST /start_session`. Ditto forwards it directly to `room.connect()` — LiveKit's server validates the token, not this codebase. The token is not generated, verified, or inspected by Vision-AI.
- **Agent Identity Check**: Within LiveKit sessions, Ditto checks `participant_id == agent_identity` before accepting audio ByteStreams or RPC calls. This prevents processing audio from unintended room participants, but is not an authentication mechanism.

### What Does Not Exist

- No API keys, Bearer tokens, OAuth, or session cookies on any endpoint
- No middleware that checks credentials on incoming requests
- No rate limiting or request throttling
- No role-based access control
- No server-side token generation or validation
- No `.env`-based secret management

### Implication

All services are designed to run on private networks or behind an external reverse proxy / API gateway that handles authentication. If exposed to the public internet as-is, any client can call any endpoint.

---

## 7. Background Jobs / Queues

### 7.1 Pre-render Clip Generation (Ditto)

**Executor:** `ThreadPoolExecutor(max_workers=1, thread_name_prefix="prerender")`
- Single-threaded to serialize GPU-intensive clip generation

**Trigger Points:**
- `POST /register` with `prerender_clips=True` (automatic after avatar registration)
- `POST /avatars/{avatar_id}/prerender` (manual trigger)

**Job Lifecycle:**
```
_start_prerender(avatar_id, image_path)
  │
  ├─ prerender_jobs[avatar_id] = {status: "pending", started: time.time()}
  │
  └─ executor.submit(_run_prerender)
       │
       ├─ Status: "setting_up" → runs skyreels/setup.sh if .setup_done missing (1hr timeout)
       ├─ Status: "running" → subprocess: python prerender_clips.py (30min timeout)
       │    └─ Generates: {avatar_id}_{idle|thinking|lookingup|lippurse|greeting}_{variant}.mp4
       ├─ Status: "done" (on success)
       └─ Status: "failed" + error message (on failure)
```

**Subprocess Command:**
```bash
PYTHONPATH={SKYREELS_PATH} {SKYREELS_PYTHON} prerender_clips.py \
  --image_path {avatar_image} \
  --output_dir {CLIPS_DIR} \
  --avatar_id {id} \
  --clips idle thinking lookingup lippurse greeting \
  --target_fps 24
```

### 7.2 Ditto Pipeline Worker Threads

**Created per video generation call** (offline pipeline in `stream_pipeline_offline.py`):

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│ audio2motion     │────►│ gpu_worker        │────►│ putback      │────►│ writer       │
│ worker           │     │ (stitch+warp+     │     │ worker       │     │ worker       │
│                  │     │  decode merged)   │     │              │     │              │
│ audio2motion_q ──┤     │ motion_stitch_q ──┤     │ putback_q ───┤     │ writer_q ────┤
└──────────────────┘     └──────────────────┘     └──────────────┘     └──────────────┘
```

- **Queue type:** `queue.Queue(maxsize=QUEUE_MAX_SIZE)`
- **Stop mechanism:** `threading.Event()` — `sdk.close()` sets event, threads drain queues and exit
- **Threads per generation:** 4 (`threading.Thread`, daemon=False)

### 7.3 Async Tasks (LiveKit Sessions)

**Created per streaming session:**

| Task | Function | Purpose | Cancellation |
|---|---|---|---|
| `publisher_task` | `_unified_publisher()` | Continuously publish video frames (idle clips or speech) at 25fps via AVSynchronizer | Cancelled in `_auto_stop_session()` |
| `audio_recv_task` | `_audio_receiver_task()` | Listen for agent audio via LiveKit ByteStream, process through Ditto pipeline | Cancelled in `_auto_stop_session()` |

**Lifecycle:** Created in `POST /start_session`, destroyed in `POST /stop_session` or on room/agent disconnect event.

### 7.4 Chatterbox TTS Executor

- WebSocket `/ws` runs model inference in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking the event loop
- Default executor (ThreadPoolExecutor) used — no custom sizing

---

## 8. Third-Party Integrations

### AI Model Providers

| Integration | Module | What It Does | Credentials |
|---|---|---|---|
| **HuggingFace Hub** | All services | Model weight downloads (SmolVLM2, SkyReels-A1-5B, DiffPoseTalk, FLAME, Ditto weights, T5-v1_1-xxl) | `HF_TOKEN` env var or `--token` flag (optional for public models) |
| **Google Chirp3 HD** | Referenced in README | Cloud TTS fallback/alternative | Not implemented in code — external to this repo |

### Real-Time Communication

| Integration | Module | What It Does | Credentials |
|---|---|---|---|
| **LiveKit** | `ditto/ditto_api.py` | WebRTC signaling, video/audio track publishing, ByteStream data channels, RPC methods | `livekit_url` + `livekit_token` (JWT) passed by client per session; no server-side LiveKit API keys in this codebase |

### GPU Cloud

| Integration | Module | What It Does | Credentials |
|---|---|---|---|
| **RunPod** | `ditto/Dockerfile`, `ditto/setup.sh` | GPU cloud deployment (A100 pods), Docker container runtime | RunPod account (external portal, no API keys in code) |
| **Hetzner** | Referenced in README | LiveKit Agent server hosting | Not configured in code |

### Open-Source Model Dependencies

| Project | Organization | Used In | Downloaded From |
|---|---|---|---|
| ditto-talkinghead | Antgroup Creative | Ditto | HuggingFace: `digital-avatar/ditto-talkinghead` |
| SkyReels-A1 | SkyworkAI | SkyReels | HuggingFace: `Skywork/SkyReels-A1` |
| Chatterbox | Resemble AI | Chatterbox | PyPI: `chatterbox-tts` |
| CogVideoX | THUDM | SkyReels (foundation) | Bundled in SkyReels-A1 weights |
| DiffPoseTalk | — | SkyReels | HuggingFace (via setup.sh) |
| FILM | DAJES | SkyReels | GitHub releases (via setup.sh) |
| pytorch3d | Facebook Research | SkyReels | Built from source or prebuilt wheel |
| GridSample3D TRT Plugin | SeanWangJS | Ditto | GitHub: `SeanWangJS/grid-sample3d-trt-plugin` |
| SmolVLM2-2.2B-Instruct | HuggingFace | SmolVLM | HuggingFace model hub (auto-download) |

### System Dependencies

| Dependency | Required By | Installed Via |
|---|---|---|
| NVIDIA CUDA 12.x | All services | Docker base image or host system |
| TensorRT 10.7.x | Ditto | `uv sync` (pyproject.toml) |
| ffmpeg | Ditto (audio merge) | `apt-get` in Dockerfile |
| libsndfile1 | Chatterbox, Ditto | `apt-get` in Dockerfile |
| cmake + make | Ditto (GridSample3D plugin) | `apt-get` in Dockerfile |

---

## 9. Deployment Architecture

### Container Strategy

Only Ditto has a Dockerfile. Chatterbox, SmolVLM, and SkyReels are run directly on the host or in manually configured environments.

### Ditto Docker Image

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# System: Python 3.10, ffmpeg, libsndfile1, build-essential, git, wget
# Package manager: uv (copied from ghcr.io/astral-sh/uv:latest)
# Dependencies: uv sync --frozen (pyproject.toml + uv.lock)
# Build step: Cython blend extension compilation
# Exposed port: 8181
# Entry: CMD ["uv", "run", "ditto_api.py"]
```

**Recommended Volume Mounts:**

| Container Path | Purpose | Persistence |
|---|---|---|
| `/workspace/avatar_cache` | Avatar feature pickle files | Required for restart survival |
| `/workspace/avatar_clips` | Pre-rendered MP4 clips | Required for clip serving |
| `/workspace/avatar_images` | Source portrait images | Required for prerendering |
| `/workspace/SkyReels-A1` | SkyReels environment + models (~15GB) | Required for clip generation |

### Service Startup Commands

```bash
# Ditto (containerized)
docker run --gpus all -p 8181:8181 \
  -v /data/avatar_cache:/workspace/avatar_cache \
  -v /data/avatar_clips:/workspace/avatar_clips \
  -v /data/avatar_images:/workspace/avatar_images \
  ditto-avatar

# Ditto (bare metal)
cd ditto && bash setup.sh && uv run uvicorn ditto_api:app --host 0.0.0.0 --port 8181

# Chatterbox
cd chatterbox && pip install -r requirements.txt && uvicorn api_server:app --host 0.0.0.0 --port 8080

# SmolVLM
cd smolvlm && pip install -r requirements.txt && uvicorn smolvlm_server:app --host 0.0.0.0 --port 8282

# SkyReels (Gradio UI)
cd skyreels && bash setup.sh && .venv/bin/python app.py
```

### Hardware Requirements

| Service | Minimum GPU | VRAM | Recommended |
|---|---|---|---|
| Ditto (TRT backend) | A100 40GB | 40GB+ | A100 80GB |
| Ditto (ONNX backend) | A100 40GB | 40GB+ | A100 80GB |
| SkyReels-A1 | A100 40GB | 40GB+ | A100 80GB |
| SmolVLM | RTX 3080 | 16GB+ | A10 24GB |
| Chatterbox | RTX 3060 | 8GB+ | RTX 3080 |

### CI/CD

**None configured.** No GitHub Actions workflows, Jenkinsfile, docker-compose, or Makefile exist in the repository.

### Build Pipeline (Manual)

```
1. ditto/setup.sh
   ├── Install uv package manager
   ├── uv sync (Python deps)
   ├── download_weights.sh (HuggingFace → checkpoints/)
   ├── cvt_onnx_to_trt.py (ONNX → TensorRT engines)
   ├── Build GridSample3D TRT plugin (cmake + make)
   └── Build Cython blend extension

2. skyreels/setup.sh
   ├── Create Python 3.12 venv
   ├── Install PyTorch 2.8 (CUDA 12.1)
   ├── Build pytorch3d from source
   ├── Download SkyReels-A1-5B model (~15GB)
   ├── Download DiffPoseTalk weights
   └── Download FILM interpolation model

3. chatterbox: pip install -r requirements.txt
4. smolvlm: pip install -r requirements.txt
```

---

## 10. Key Environment Variables

### Service Configuration (Ditto)

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `DITTO_PATH` | `./ditto-talkinghead` | No | Path to Ditto inference engine directory |
| `DITTO_BACKEND` | `onnx` | No | Inference backend: `trt` (TensorRT) or `onnx` (ONNX Runtime) |
| `AVATAR_CACHE_DIR` | `/workspace/avatar_cache` | No | Directory for avatar feature pickle caches |
| `AVATAR_CLIPS_DIR` | `/workspace/avatar_clips` | No | Directory for pre-rendered expression clips |
| `AVATAR_IMAGES_DIR` | `/workspace/avatar_images` | No | Directory for avatar source images |
| `TTS_URL` | `http://localhost:8282/tts/stream` | No | Chatterbox TTS endpoint URL (used by `/generate_from_text`) |

### SkyReels Integration (Ditto)

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `SKYREELS_PATH` | `/workspace/SkyReels-A1` | No | Path to SkyReels-A1 installation |
| `SKYREELS_PYTHON` | `.venv/bin/python` | No | Python interpreter in SkyReels virtual environment |
| `PRERENDER_SCRIPT` | (derived from `SKYREELS_PATH`) | No | Path to `prerender_clips.py` |

### Build & System

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `LD_LIBRARY_PATH` | (system) | For TRT backend | Must include TensorRT library paths |
| `CUDA_HOME` | (system) | For pytorch3d build | CUDA toolkit installation path |
| `HF_TOKEN` | (none) | For private models | HuggingFace authentication token |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | No | Set to `1` for parallel HuggingFace downloads |

### Rendering & Reproducibility

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `PYOPENGL_PLATFORM` | (system) | For headless rendering | Set to `egl` in SkyReels renderer |
| `PYTHONHASHSEED` | (random) | No | Set to fixed value for reproducible inference |
| `GRADIO_TEMP_DIR` | (system temp) | No | Set to `tmp` in SkyReels Gradio app |

**Note:** No `.env` or `.env.example` file exists. All variables have hardcoded defaults and are optional for basic operation.

---

## 11. Real-Time / Event Flows

### 11.1 LiveKit WebRTC Streaming (Ditto)

The primary real-time path. An AI agent sends audio to Ditto via LiveKit DataStream; Ditto generates lip-synced video frames and publishes them back to the LiveKit room as a WebRTC video/audio track.

**Protocol:**

```
AI Agent                        Ditto Avatar                     Client Browser
   │                               │                                │
   │── POST /start_session ───────►│                                │
   │   {avatar_id, livekit_url,    │                                │
   │    livekit_token,             │                                │
   │    agent_identity}            │                                │
   │                               │── room.connect() ─────────────►│ LiveKit
   │                               │── publish video track (VP8) ──►│ Server
   │                               │── publish audio track (24kHz)─►│
   │                               │                                │
   │◄── {session_id, status} ──────│                                │
   │                               │                                │
   │                               │   _unified_publisher loop:     │
   │                               │   ┌─ If no speech audio:      │
   │                               │   │  Play idle/thinking clip   │──► WebRTC ──► Browser
   │                               │   │  + silent audio            │
   │                               │   └─ Loop at 25fps             │
   │                               │                                │
   │── ByteStream ────────────────►│   _audio_receiver_task:        │
   │   topic: "lk.audio_stream"   │   ┌─ Receive PCM audio chunk  │
   │   (raw PCM bytes)            │   │  Normalize to 24kHz mono  │
   │                               │   │  Resample to 16kHz        │
   │                               │   │  Extract HuBERT features  │
   │                               │   │  Run Ditto pipeline:      │
   │                               │   │  audio2motion → stitch    │
   │                               │   │  → warp → decode → putback│
   │                               │   │  Push frames to publisher │
   │                               │   └─────────────────────────  │
   │                               │                                │
   │                               │   _unified_publisher:          │
   │                               │   ┌─ Drain output_queue       │
   │                               │   │  Push frame via AVSync   ─┤──► WebRTC ──► Browser
   │                               │   │  Push audio (delayed 5   ─┤──► WebRTC ──► Browser
   │                               │   │   frames = 200ms)         │
   │                               │   │  Pace at 25fps precise   │
   │                               │   └─────────────────────────  │
   │                               │                                │
   │── RPC "lk.clear_buffer" ─────►│   Interrupt:                   │
   │                               │   Set _speech_interrupted      │
   │                               │   Discard pending frames       │
   │                               │   Resume idle clips            │
   │                               │                                │
   │◄── RPC "lk.playback_finished"│   After segment completes:     │
   │   {playback_position: float, │                                │
   │    interrupted: bool}         │                                │
   │                               │                                │
   │── POST /stop_session ────────►│   Cleanup:                     │
   │                               │   Cancel async tasks           │
   │                               │   Close AVSync                 │
   │                               │   Disconnect room              │
```

**LiveKit DataStream Topics:**

| Topic | Direction | Payload | Purpose |
|---|---|---|---|
| `lk.audio_stream` | Agent → Ditto | Raw PCM bytes (variable sample rate, mono/stereo) | Speech audio for lip-sync |

**LiveKit RPC Methods:**

| Method | Direction | Payload | Response | Purpose |
|---|---|---|---|---|
| `lk.clear_buffer` | Agent → Ditto | (none) | `"ok"` or `"reject"` | Interrupt current speech, discard frames |
| `lk.playback_finished` | Ditto → Agent | `{"playback_position": float, "interrupted": bool}` | (ack) | Report segment completion status |

**Published Tracks:**

| Track | Codec | Config | Content |
|---|---|---|---|
| `ditto-avatar-video` | VP8 | 2Mbps max bitrate, 25fps | Avatar face frames (resolution from source image) |
| `ditto-avatar-audio` | Opus (LiveKit default) | 24kHz mono, 100ms queue | Agent speech audio (passthrough with 200ms delay) |

**Audio Synchronization:**
- Audio is delayed by `AUDIO_DELAY_FRAMES = 5` frames (200ms at 25fps) relative to video
- This hides the LMDM diffusion model ramp-up latency
- `AVSynchronizer` handles RTP timestamp generation and A/V sync
- Frame pacing uses `time.perf_counter()` with busy-wait for ±1ms precision

**Auto-Disconnect Handlers:**
- `room.on("participant_disconnected")` → stops session if agent leaves
- `room.on("disconnected")` → stops session if room connection drops

### 11.2 Chatterbox WebSocket TTS

Session-based streaming TTS with voice cloning support.

**Connection Lifecycle:**

```
Client                          Chatterbox Server
   │                               │
   │── WS connect /ws ────────────►│
   │                               │  Create session_id, temp dir
   │                               │
   │── {"type": "init",           │
   │    "voice_prompt": "base64"} ►│  Decode WAV, save to temp
   │                               │  model.prepare_conditionals()
   │◄── {"type": "ready",         │
   │     "session_id": "...",      │
   │     "voice_cloning": true,    │
   │     "model": "turbo",        │
   │     "supports": ["[laugh]",  │
   │      "[cough]", "[sigh]",    │
   │      ...]}                    │
   │                               │
   │── {"type": "tts",            │
   │    "text": "Hello!",          │
   │    "temperature": 0.8,        │
   │    "top_p": 0.95}       ─────►│
   │                               │  model.generate() in executor
   │◄── {"type": "processing"} ────│
   │                               │  Convert tensor → WAV bytes
   │◄── {"type": "audio",         │
   │     "data": "base64_wav",     │
   │     "duration": 1.2,          │
   │     "process_time": 0.8,      │
   │     "size_kb": 48.5}     ────│
   │                               │
   │── {"type": "ping"} ──────────►│
   │◄── {"type": "pong"} ─────────│
   │                               │
   │── {"type": "close"} ─────────►│  Cleanup: delete temp dir
   │◄── WS close ─────────────────│
```

**Supported Paralinguistic Tags:** `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[clear throat]`

### 11.3 Ditto Test WebSocket

Debug-only endpoint for testing without LiveKit.

**Endpoint:** `WS /ws/test_frames?avatar_id={id}&duration={seconds}`

**Flow:**
1. Server generates sine-wave test audio (440Hz)
2. Runs Ditto offline pipeline
3. Streams results as WebSocket messages:
   - `{"type": "frame", "data": "<base64_jpeg>"}` — video frames
   - `{"type": "audio", "data": "<base64_wav>"}` — audio chunk

### 11.4 No Other Event Systems

There are no:
- Server-Sent Events (SSE)
- Pub/sub message brokers (Redis, RabbitMQ, Kafka)
- Event queues beyond Python's `queue.Queue` and `asyncio.Queue`
- Broadcasting mechanisms beyond LiveKit's WebRTC room

---

## 12. Server Access

### What Exists in Code

**Listening Interfaces:**

| Service | Host | Port | Configured In |
|---|---|---|---|
| Ditto | `0.0.0.0` | `8181` | `ditto/ditto_api.py` (hardcoded at bottom of file) |
| Chatterbox | `0.0.0.0` | `8081` | `chatterbox/api_server.py` (argparse, default 8081) |
| SmolVLM | `0.0.0.0` | `8282` | `smolvlm/smolvlm_server.py` (hardcoded at bottom of file) |
| SkyReels Gradio | `0.0.0.0` | `7860` (Gradio default) | `skyreels/app.py` (`server_name="0.0.0.0"`) |

**Docker Port Exposure:**
- Ditto Dockerfile: `EXPOSE 8181` (only Ditto is containerized)

### What Does Not Exist in Code

- No SSH configuration files, keys, or authorized_keys setup
- No server IP addresses or hostnames hardcoded (except LiveKit URL format `wss://...` in comments)
- No firewall rules (iptables, ufw, security groups)
- No reverse proxy configuration (nginx, Caddy, Traefik)
- No TLS/SSL certificate configuration (all services serve plain HTTP)
- No deployment user setup or privilege management
- No bastion host or jump server configuration
- No VPN or network segmentation rules

### External References (from README/comments only)

| Reference | Context | Not Configured In Code |
|---|---|---|
| RunPod | GPU cloud platform | Deployment managed via RunPod web portal |
| Hetzner | Mentioned for LiveKit Agent server | No server config in repo |
| LiveKit Server | WebRTC SFU | URL provided by client at session start (`wss://...`) |

### Access Control Summary

All services bind to `0.0.0.0` with no authentication. Network-level access control (firewalls, security groups, VPN) must be configured externally at the infrastructure layer. The codebase assumes it runs on a trusted private network.
