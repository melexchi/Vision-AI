# Vision-AI Codebase Audit

**Generated:** 2026-04-01
**Repository:** Self-hosted GPU inference stack for real-time AI talking avatars

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Tech Stack](#2-tech-stack)
3. [Data Flow](#3-data-flow)
4. [Database Models & State Management](#4-database-models--state-management)
5. [API Routes & Endpoints](#5-api-routes--endpoints)
6. [Shared Utilities & Common Patterns](#6-shared-utilities--common-patterns)
7. [Test Setup](#7-test-setup)
8. [Build, Deploy & CI/CD](#8-build-deploy--cicd)
9. [Environment Variables & Config](#9-environment-variables--config)
10. [Third-Party Integrations](#10-third-party-integrations)
11. [Dead Code & Cleanup Opportunities](#11-dead-code--cleanup-opportunities)

---

## 1. Directory Structure

| Folder/File | Description |
|---|---|
| `ditto/` | Real-time lip-sync avatar streaming service (Port 8181) — TensorRT/PyTorch pipeline with LiveKit WebRTC |
| `ditto/ditto-talkinghead/` | Core Ditto inference engine: 54 Python files with atomic pipeline components, neural models, and TRT utils |
| `skyreels/` | Expression clip generation using SkyReels-A1 image-to-video diffusion (CogVideoX-based, 8B params) |
| `skyreels/skyreels_a1/` | Main SkyReels-A1 model pipeline — 3D transformer, DDIM solver, landmark preprocessing |
| `skyreels/diffposetalk/` | Audio-to-FLAME-motion component — converts speech to 3D face motion parameters |
| `skyreels/eval/` | Evaluation metrics — ArcFace similarity, expression quality, pose accuracy |
| `skyreels/assets/` | Example media: driving audio/video clips, reference portrait images |
| `chatterbox/` | Neural TTS service (Port 8080) — voice cloning, multilingual support via Chatterbox library |
| `smolvlm/` | Camera vision service (Port 8282) — SmolVLM2-2.2B-Instruct for image understanding |
| `.claude/` | Claude Code local configuration |
| `README.md` | Project documentation with architecture diagrams and hardware requirements |
| `.gitignore` | Ignores model files (*.pt, *.pth, *.onnx, *.engine, *.pkl, *.bin, *.safetensors), Python cache, .env |

### Ditto Internal Structure

```
ditto/
├── ditto_api.py              # FastAPI server — all REST/WebSocket endpoints (79K)
├── prerender_clips.py        # SkyReels-A1 clip generation (17K)
├── Dockerfile                # RunPod GPU deployment container
├── setup.sh                  # Installation & dependency setup
├── download_weights.sh       # Model weight downloader from HuggingFace
├── requirements.txt
├── pyproject.toml
├── test_viewer.html          # HTML debug viewer
├── uv.lock                   # Dependency lock (357K)
└── ditto-talkinghead/
    ├── core/
    │   ├── atomic_components/ # 12 modular pipeline stages (audio2motion, avatar_registrar, wav2feat, etc.)
    │   ├── aux_models/        # Face detection, landmarks, HuBERT (InsightFace, MediaPipe, BlazeFace)
    │   ├── models/            # Core neural nets (LMDM, decoder, appearance/motion extractors, warp/stitch)
    │   └── utils/             # Crop, eye info, masks, TRT wrapper, Cython blend extension
    ├── scripts/
    │   └── cvt_onnx_to_trt.py # ONNX → TensorRT engine converter
    ├── inference.py
    ├── stream_pipeline_online.py
    ├── stream_pipeline_offline.py
    └── environment.yaml       # Conda environment spec
```

### SkyReels Internal Structure

```
skyreels/
├── app.py                     # Gradio web UI
├── inference.py               # Image-to-video inference (video driving)
├── inference_audio.py         # Audio-to-video inference
├── inference_audio_long_video.py
├── inference_long_video.py
├── setup.sh
├── requirements.txt
├── skyreels_a1/
│   ├── skyreels_a1_i2v_pipeline.py
│   ├── skyreels_a1_i2v_long_pipeline.py
│   ├── models/transformer3d.py  # 3D transformer for video diffusion (36K)
│   ├── ddim_solver.py
│   └── src/
│       ├── frame_interpolation.py
│       ├── smirk_encoder.py
│       ├── renderer.py
│       ├── FLAME/               # Basel parametric 3D face model
│       └── media_pipe/          # Face detection + landmark extraction
├── diffposetalk/
│   ├── diffposetalk.py
│   ├── diff_talking_head.py
│   ├── hubert.py, wav2vec2.py
│   └── utils/                   # Rotation conversions, rendering, media
├── eval/                        # ArcFace, expression, pose scoring
└── scripts/demo.py              # Pipeline initialization helpers
```

---

## 2. Tech Stack

### Core ML Frameworks

| Technology | Version | Purpose |
|---|---|---|
| PyTorch | 2.8.0 (Ditto), 2.2.2+cu118 (SkyReels) | Deep learning framework |
| TensorRT | 10.7.x (CUDA 12) | Ultra-low-latency GPU inference for Ditto pipeline |
| ONNX Runtime GPU | >=1.17.0 | ONNX model inference (fallback backend) |
| HuggingFace Transformers | >=4.49.0 | SmolVLM2, T5 encoder, model loading |
| HuggingFace Diffusers | 0.32.2 | CogVideoX-based video diffusion (SkyReels-A1) |
| CUDA | 12.1 / 12.8 | GPU compute |

### Web Frameworks & Real-time

| Technology | Purpose |
|---|---|
| FastAPI + Uvicorn | HTTP/WebSocket API servers (all 4 services) |
| LiveKit SDK | WebRTC real-time video/audio streaming (Ditto) |
| Gradio | Interactive ML demo UI (SkyReels) |
| Pydantic | Request/response validation |
| httpx | Async HTTP client |

### Computer Vision & Audio

| Technology | Purpose |
|---|---|
| OpenCV | Image/video processing |
| MediaPipe | Face mesh, pose, landmarks |
| InsightFace | Face detection + recognition |
| Librosa | Audio feature extraction |
| torchaudio | Audio processing |
| Chatterbox TTS | Neural TTS with voice cloning |
| pytorch3d | 3D vision/graphics (built from source) |
| FLAME | Parametric 3D face model |

### Infrastructure & Deployment

| Technology | Purpose |
|---|---|
| Docker | Containerization (nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04) |
| RunPod | GPU cloud deployment platform |
| uv | Python package manager (primary) |
| Conda | Environment management (legacy, for ditto-talkinghead) |
| ffmpeg | Video codec (system dependency) |

### Key AI Models

| Model | Params | Service | Purpose |
|---|---|---|---|
| Ditto TalkingHead | Multi-model | Ditto | Real-time lip-sync (HuBERT + LMDM + Decoder) |
| SkyReels-A1 | 8B | SkyReels | Portrait animation via diffusion transformers |
| SmolVLM2-2.2B-Instruct | 2.2B | SmolVLM | Vision-language understanding |
| Chatterbox Multilingual/Turbo | Varies | Chatterbox | Neural TTS with voice cloning |
| DiffPoseTalk | — | SkyReels | Audio-to-FLAME motion generation |
| FILM | — | SkyReels | Frame interpolation |

---

## 3. Data Flow

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     VISION-AI SYSTEM                              │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────┐│
│  │ Chatterbox   │  │  Ditto       │  │ SkyReels  │  │ SmolVLM  ││
│  │ TTS :8080    │  │  Avatar:8181 │  │ (Gradio)  │  │ VLM:8282 ││
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  └────┬─────┘│
│         │                 │                │              │       │
│         └────────┬────────┘                │              │       │
│                  │ (TTS audio feeds Ditto)  │              │       │
│                  │                          │              │       │
│                  ├──────────────────────────┘              │       │
│                  │ (SkyReels generates idle/expr clips)    │       │
│                  │                                          │       │
│         ┌────────▼────────┐                                │       │
│         │  LiveKit WebRTC │◄───── Client Browser ──────────┘       │
│         │  (25fps stream) │                                        │
│         └─────────────────┘                                        │
└──────────────────────────────────────────────────────────────────┘
```

### Ditto Request Lifecycle (Primary Service)

#### Avatar Registration Flow
```
POST /register (image_base64 or image_path)
  → Face detection (InsightFace)
  → Landmark extraction (106 + 203 points)
  → Appearance feature extraction
  → Motion parameter extraction
  → Cache source_info to memory + disk (pickle)
  → Optionally trigger SkyReels prerendering (background thread)
  → Return avatar_id
```

#### Offline Video Generation Flow
```
POST /generate (avatar_id + audio)
  → Load source_info from avatar_cache
  → Audio → 16kHz resampling
  → HuBERT feature extraction (wav2feat)
  → Multi-threaded pipeline:
      audio2motion (LMDM, 5 DDIM steps) → motion_stitch → warp_f3d → decode_f3d → putback → writer
  → ffmpeg audio merge
  → Return MP4
```

#### Real-time Streaming Flow (LiveKit)
```
POST /start_session (avatar_id + LiveKit credentials)
  → Connect to LiveKit room
  → Publish video + audio tracks
  → Launch async tasks:
      1. _unified_publisher: consume audio → run Ditto pipeline → push frames at 25fps via AVSynchronizer
      2. _audio_receiver_task: listen for agent audio via LiveKit DataStream
  → On disconnect: auto-stop, cleanup resources
```

### Chatterbox TTS Flow
```
POST /api/tts (text + optional voice_prompt)
  → Load Multilingual or Turbo model (lazy)
  → Generate waveform (with optional voice cloning)
  → Return WAV bytes

WebSocket /ws
  → "init": load voice embeddings from base64 WAV
  → "tts": generate audio, return base64 WAV
  → Supports paralinguistic tags: [laugh], [cough], [sigh], etc.
```

### SmolVLM Flow
```
POST /v1/query (image_base64/url + prompt)
  → Lazy load SmolVLM2-2.2B-Instruct
  → Process image + text prompt
  → model.generate(max_new_tokens=512)
  → Return text response
```

### Inter-Service Communication
- **Ditto → Chatterbox**: HTTP call to `TTS_URL` (default `http://localhost:8282/tts/stream`) for text-to-speech in `/generate_from_text`
- **Ditto → SkyReels**: Subprocess call to `prerender_clips.py` for idle/thinking/greeting clip generation
- **Client → Ditto**: LiveKit WebRTC for real-time streaming; REST for offline generation
- **Client → SmolVLM**: REST for camera frame analysis

---

## 4. Database Models & State Management

**This repository uses NO SQL database.** All state is managed via in-memory dictionaries and disk caches.

### Ditto State

| Store | Type | Key | Contents |
|---|---|---|---|
| `avatar_cache` | In-memory dict | `avatar_id` | `source_info` — per-frame motion params, RGB frames, landmarks, eye tracking, appearance features (~9MB per avatar) |
| `{CACHE_DIR}/{id}.pkl` | Disk (pickle) | `avatar_id` | Serialized `source_info` for persistence across restarts |
| `{IMAGES_DIR}/{id}.png` | Disk (PNG) | `avatar_id` | Original source portrait image |
| `{CLIPS_DIR}/{id}_{type}_{variant}.mp4` | Disk (MP4) | `avatar_id` + clip type | Pre-rendered expression clips (5 types: idle, thinking, lookingup, lippurse, greeting; multiple variants each) |
| `active_sessions` | In-memory dict | `session_id` (UUID) | LiveKit room, video/audio sources, audio buffers, async task handles, AVSynchronizer |
| `prerender_jobs` | In-memory dict | `avatar_id` | Background clip rendering status and metadata |

### Chatterbox State

| Store | Type | Contents |
|---|---|---|
| `tts_model` / `multilingual_model` / `turbo_model` | Global variables | Lazy-loaded model instances |
| `{TEMP_DIR}/chatterbox_api/{session_id}/` | Disk (ephemeral) | WebSocket session voice prompts (cleaned up on disconnect) |

### SmolVLM State

| Store | Type | Contents |
|---|---|---|
| `vlm_model` | Global variable | Lazy-loaded AutoModelForImageTextToText |
| `vlm_processor` | Global variable | Lazy-loaded AutoProcessor |

---

## 5. API Routes & Endpoints

### Chatterbox (Port 8080)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model load status |
| POST | `/api/tts` | Text-to-speech (Form: text, language, voice_prompt, exaggeration, cfg_weight, model_type) → WAV |
| WS | `/ws` | Streaming TTS with session-based voice cloning |

### Ditto (Port 8181)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status, cached avatar count, config |
| POST | `/register` | Register avatar from image (base64/path), optionally trigger clip prerendering |
| POST | `/generate` | Generate video from avatar_id + audio (base64/path) → MP4 |
| POST | `/generate_from_text` | Generate video from avatar_id + text (calls Chatterbox internally) → MP4 |
| POST | `/generate_frames` | Stream raw JPEG frames from PCM audio (Headers: X-Avatar-Id, X-Sample-Rate) |
| POST | `/start_session` | Start real-time LiveKit streaming session |
| POST | `/stop_session/{session_id}` | Stop streaming session |
| GET | `/avatars` | List all registered avatars with status |
| GET | `/avatars/{avatar_id}/status` | Single avatar inference/clip readiness |
| GET | `/avatars/{avatar_id}/resolution` | Output video dimensions |
| DELETE | `/avatars/{avatar_id}` | Evict avatar (memory + disk + clips) |
| GET | `/clips/{avatar_id}/{clip_type}` | Serve pre-rendered expression clip (random variant) |
| POST | `/avatars/{avatar_id}/prerender` | Manually trigger clip pre-rendering |
| GET | `/test_viewer` | HTML test page |
| WS | `/ws/test_frames` | Test WebSocket (sine-wave audio → JPEG frames) |

### SmolVLM (Port 8282)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status |
| POST | `/v1/query` | Query with image (base64/URL) + text prompt → text response |
| POST | `/v1/query/upload` | Query with uploaded image file + prompt → text response |
| POST | `/v1/unload` | Free VRAM (delete model, empty CUDA cache) |

### SkyReels

No REST API. Runs as a Gradio web UI (`app.py`) or via CLI inference scripts.

---

## 6. Shared Utilities & Common Patterns

### Reusable Utility Modules

**Ditto Core Utils** (`ditto/ditto-talkinghead/core/utils/`):
- `tensorrt_utils.py` — TRTWrapper, DynamicShapeOutputAllocator, CUDA error checking
- `crop.py` — Face cropping, landmark parsing (101/106/203/68/5/9 point formats), affine transforms, paste-back
- `load_model.py` — Unified model loader dispatching to ONNX/TRT/PyTorch based on file extension
- `eye_info.py` — MediaPipe eye landmark indices and attribute extraction
- `get_mask.py` — Face mask generation for blending
- `blend/` — Cython/C-accelerated image blending (blend.pyx + blend_impl.c)

**SkyReels Utils** (`skyreels/diffposetalk/utils/`):
- `rotation_conversions.py` — Rodrigues, quaternion, rotation matrix conversions
- `renderer.py` — 3D mesh rendering via pyrender/trimesh
- `media.py` — Media file handling
- `common.py` — Frame processing, normalization, motion smoothing

**SkyReels A1 Utils** (`skyreels/skyreels_a1/src/utils/`):
- `mediapipe_utils.py` — MediaPipe wrapper for face landmark extraction

### Cross-Codebase Patterns

| Pattern | Usage |
|---|---|
| **Lazy model loading** | All 4 services load models on first request, not at startup (except Ditto warm-load) |
| **FastAPI + CORS** | All HTTP services use FastAPI with permissive CORS middleware |
| **Pydantic validation** | Request models with Optional fields and sensible defaults |
| **Async/WebSocket** | Ditto and Chatterbox support WebSocket for streaming |
| **CUDA device management** | All services use `torch.device("cuda")` with consistent patterns |
| **Pickle caching** | Ditto serializes heavy feature dicts to disk for persistence |
| **Background tasks** | ThreadPoolExecutor for clip prerendering; asyncio tasks for streaming |
| **Health endpoints** | Every service exposes `GET /health` |
| **pathlib.Path** | Consistent path handling throughout |

### Inter-Service Dependencies

- **Ditto → SkyReels**: Imports `CogVideoXTransformer3DModel`, `SkyReelsA1ImagePoseToVideoPipeline`, `FaceAnimationProcessor`, `LMKExtractor` in `prerender_clips.py`
- **Ditto → Chatterbox**: HTTP call for TTS in `/generate_from_text`
- No other cross-service code dependencies; each service is otherwise self-contained

---

## 7. Test Setup

### Current State: Minimal / No Automated Tests

- **No test framework configured** — no pytest, unittest, nose, or similar
- **No test infrastructure** — no `conftest.py`, `pytest.ini`, `setup.cfg`, `.coveragerc`, `tox.ini`
- **No CI/CD test workflows** — no `.github/workflows/` directory

### Existing Test Files

| File | Type | Purpose |
|---|---|---|
| `skyreels/skyreels_a1/src/lmk3d_test.py` | Manual script | Standalone face animation processor test — run manually with `--source_image`, `--driving_video`, `--output_path` args |
| `ditto/test_viewer.html` | HTML page | Browser-based WebSocket frame viewer for manual debugging |
| `skyreels/test_viewer.html` | HTML page | Browser-based test viewer |
| `ditto/ditto_api.py` `/ws/test_frames` | WebSocket endpoint | Generates sine-wave test audio → streams JPEG frames for integration testing |

### How to Run (Manual Only)

```bash
# SkyReels landmark test
cd skyreels/skyreels_a1/src/
python lmk3d_test.py --source_image <path> --driving_video <path> --output_path <output.mp4>

# Ditto WebSocket test
# Start server, then open test_viewer.html in browser
```

---

## 8. Build, Deploy & CI/CD

### Build Commands

**Ditto (most complex build):**
```bash
# Full setup (idempotent)
cd ditto && bash setup.sh

# What setup.sh does:
# 1. Install uv package manager
# 2. uv sync (install Python deps from pyproject.toml + uv.lock)
# 3. Download model weights (download_weights.sh → HuggingFace)
# 4. Build TRT engines from ONNX (cvt_onnx_to_trt.py)
# 5. Compile GridSample3D TRT plugin (cmake + make)
# 6. Compile Cython blend extension
```

**SkyReels:**
```bash
cd skyreels && bash setup.sh

# What setup.sh does:
# 1. Create Python 3.12 venv
# 2. Install PyTorch (CUDA 12.1)
# 3. Build pytorch3d from source (complex C++ build)
# 4. Download SkyReels-A1-5B model (~15GB from HuggingFace)
# 5. Download DiffPoseTalk + FILM weights
# 6. Creates .setup_done marker (idempotent)
```

**Chatterbox & SmolVLM:**
```bash
pip install -r requirements.txt
# Models lazy-loaded at runtime from HuggingFace
```

### Docker Deployment

**Ditto Dockerfile** (`ditto/Dockerfile`):
- Base: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- Installs Python 3.10, uv, system deps (ffmpeg, libsndfile1, build-essential)
- Caches dependency layers
- Builds Cython blend extension
- Exposes port 8181
- Target: RunPod GPU cloud

### CI/CD

**None configured.** No GitHub Actions, Jenkinsfile, docker-compose, or Makefile found.

### Running Services

```bash
# Ditto
cd ditto && uv run uvicorn ditto_api:app --host 0.0.0.0 --port 8181

# Chatterbox
cd chatterbox && uvicorn api_server:app --host 0.0.0.0 --port 8080

# SmolVLM
cd smolvlm && uvicorn smolvlm_server:app --host 0.0.0.0 --port 8282

# SkyReels (Gradio UI)
cd skyreels && python app.py
```

---

## 9. Environment Variables & Config

### Environment Variables

**Ditto (`ditto_api.py`):**

| Variable | Default | Purpose |
|---|---|---|
| `DITTO_PATH` | `./ditto-talkinghead` | Path to Ditto engine |
| `AVATAR_CACHE_DIR` | `/workspace/avatar_cache` | Avatar feature cache directory |
| `AVATAR_CLIPS_DIR` | `/workspace/avatar_clips` | Pre-rendered clip storage |
| `AVATAR_IMAGES_DIR` | `/workspace/avatar_images` | Avatar source images |
| `SKYREELS_PATH` | `/workspace/SkyReels-A1` | Path to SkyReels-A1 installation |
| `SKYREELS_PYTHON` | `.venv/bin/python` | Python interpreter for SkyReels venv |
| `PRERENDER_SCRIPT` | (derived) | Path to prerender_clips.py |
| `TTS_URL` | `http://localhost:8282/tts/stream` | Chatterbox TTS endpoint |
| `DITTO_BACKEND` | `onnx` | Inference backend: `trt` or `onnx` |

**Build/System:**

| Variable | Purpose |
|---|---|
| `LD_LIBRARY_PATH` | TensorRT library paths |
| `CUDA_HOME` | CUDA toolkit path (pytorch3d build) |
| `HF_HUB_ENABLE_HF_TRANSFER` | Enable parallel HuggingFace downloads |
| `PYOPENGL_PLATFORM` | OpenGL backend (`egl`) for headless rendering |
| `PYTHONHASHSEED` | Reproducibility seed |
| `GRADIO_TEMP_DIR` | Gradio temp directory (`tmp`) |

### Config Files

| File | Purpose |
|---|---|
| `ditto/pyproject.toml` | Ditto dependencies + custom PyTorch/NVIDIA pip indices |
| `smolvlm/pyproject.toml` | SmolVLM dependencies |
| `ditto/uv.lock` | Reproducible dependency lock (2278 lines) |
| `ditto/ditto-talkinghead/environment.yaml` | Conda environment spec (legacy) |
| `ditto/requirements.txt` | Ditto pip dependencies |
| `skyreels/requirements.txt` | SkyReels pip dependencies |
| `chatterbox/requirements.txt` | Chatterbox pip dependencies |
| `smolvlm/requirements.txt` | SmolVLM pip dependencies |
| `.claude/settings.local.json` | Claude Code local settings |

**No `.env` or `.env.example` files exist.** All config uses environment variables with hardcoded defaults.

---

## 10. Third-Party Integrations

### AI/ML Model Sources

| Source | Models | Access |
|---|---|---|
| HuggingFace Hub | SmolVLM2-2.2B, SkyReels-A1-5B, DiffPoseTalk, FLAME, Ditto weights, T5-v1_1-xxl | Open-source downloads |
| Google Chirp3 HD | Cloud TTS | External API (mentioned as fallback/alternative) |

### Real-time Communication

| Service | Purpose |
|---|---|
| LiveKit | WebRTC signaling & media server for real-time avatar streaming |

### GPU Cloud

| Service | Purpose |
|---|---|
| RunPod | GPU cloud deployment (A100 80GB target) |
| Hetzner | LiveKit Agent server hosting (mentioned in docs) |

### Open-Source Model Origins

| Project | Organization | Used For |
|---|---|---|
| ditto-talkinghead | Antgroup Creative | Base lip-sync engine |
| SkyReels-A1 | SkyworkAI | Portrait animation diffusion |
| Chatterbox | Resemble AI | Neural TTS |
| CogVideoX | THUDM | Video diffusion foundation |
| DiffPoseTalk | — | Audio-to-motion |
| FILM | DAJES | Frame interpolation |
| pytorch3d | Facebook Research | 3D vision |

### Custom Plugins

| Plugin | Source | Purpose |
|---|---|---|
| GridSample3D TRT | github.com/SeanWangJS/grid-sample3d-trt-plugin | Custom TensorRT plugin for warp network (critical for real-time latency) |

---

## 11. Dead Code & Cleanup Opportunities

### Deprecated Code

| Location | Issue |
|---|---|
| `ditto/ditto_api.py:59` | `streaming_sdk = None  # DEPRECATED: kept for backwards compat, unused` — can be removed |

### Commented-Out Code

| File | Lines | Description |
|---|---|---|
| `ditto/ditto-talkinghead/core/models/modules/convnextv2.py:9` | Import replaced inline | `# from timm.models.layers import trunc_normal_, DropPath` |
| `ditto/ditto-talkinghead/core/utils/load_model.py:43-45` | Alternative imports | Commented reference implementations |
| `skyreels/diffposetalk/utils/renderer.py:8-9,12,98` | PYOPENGL_PLATFORM option, unused mesh import, workaround method | Legacy rendering alternatives |
| `skyreels/skyreels_a1/models/transformer3d.py:32,108` | Alternative diffusers imports, positional embedding | Dead code in transformer |
| `skyreels/skyreels_a1/skyreels_a1_i2v_pipeline.py` | TODO comments | `# 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline` |

### Legacy/Backward Compatibility Code

| Location | Issue |
|---|---|
| `ditto/ditto-talkinghead/stream_pipeline_offline.py:83` | `# Support legacy kwarg N_d= from callers` — can be removed if no callers use old kwarg |
| `ditto/ditto-talkinghead/core/utils/tensorrt_utils.py` | PyTorch fallback code for when TRT unavailable — may be intentional |

### Potential Improvements

| Area | Recommendation |
|---|---|
| **No `.env.example`** | Create `.env.example` documenting all environment variables with defaults |
| **No tests** | Add at minimum: API endpoint smoke tests, model loading tests, audio processing unit tests |
| **No CI/CD** | Add GitHub Actions for linting, type checking, and basic tests |
| **No docker-compose** | Create `docker-compose.yml` to orchestrate all 4 services together |
| **No type hints** | API server files lack comprehensive type annotations |
| **Conda + uv overlap** | `environment.yaml` appears to be legacy; consolidate on uv |
| **Port mismatch** | `TTS_URL` defaults to port 8282 but Chatterbox README says port 8080 — verify intended port |
| **CORS wildcard** | All services use `allow_origins=["*"]` — restrict in production |
| **No rate limiting** | API servers have no request throttling |
| **No auth** | No authentication on any endpoint |

### Unused Dependencies

No obviously unused pip dependencies were found — all declared dependencies map to active imports. The `environment.yaml` conda spec may be fully redundant given the `pyproject.toml` + `uv.lock` setup.

---

## Hardware Requirements

| Service | Minimum GPU | Recommended |
|---|---|---|
| Ditto (TRT) | A100 40GB | A100 80GB |
| SkyReels | A100 40GB | A100 80GB |
| SmolVLM | RTX 3080 16GB | A10 24GB |
| Chatterbox | RTX 3060 8GB | RTX 3080 |

---

## Summary Statistics

| Metric | Value |
|---|---|
| Total Python source files | ~100+ |
| Services | 4 (Ditto, SkyReels, Chatterbox, SmolVLM) |
| API endpoints | 21 (REST) + 2 (WebSocket) |
| Database | None (in-memory + disk cache) |
| Test coverage | 0% (no automated tests) |
| CI/CD pipelines | 0 |
| Environment variables | 11 |
| Docker images | 1 (Ditto only) |
| Total model parameters | ~10B+ across all services |
| Target latency | ~200ms (Ditto real-time) |
| Target throughput | 25fps video streaming |
