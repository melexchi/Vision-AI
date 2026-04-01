# Vision-AI Code Patterns & Style Guide

**Last updated:** 2026-04-01  
**Purpose:** Reference for all future code contributions. Each pattern is sourced from an actual file in this codebase.

---

## Table of Contents

1. [API Route Structure](#1-api-route-structure)
2. [Data Access Pattern](#2-data-access-pattern)
3. [Error Handling](#3-error-handling)
4. [Authentication & Middleware](#4-authentication--middleware)
5. [Environment Variables & Config](#5-environment-variables--config)
6. [Module / Feature Organization](#6-module--feature-organization)
7. [Test Patterns](#7-test-patterns)
8. [Background Jobs / Queue Tasks](#8-background-jobs--queue-tasks)
9. [Frontend Components](#9-frontend-components)
10. [API Response Format](#10-api-response-format)
11. [Naming Conventions](#11-naming-conventions)
12. [Import Patterns](#12-import-patterns)

---

## 1. API Route Structure

All services use **FastAPI** with async route handlers. Routes follow a consistent shape: decorator → Pydantic request model → input validation → business logic → response dict or `Response` object, wrapped in a top-level try/except.

**Source:** `ditto/ditto_api.py` (lines ~962–998)

```python
@app.post("/register")
async def register_avatar(request: RegisterAvatarRequest):
    """Register an avatar image. Caches features for instant reuse."""
    sdk = load_ditto()

    try:
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            temp_path = "/tmp/avatar_temp.png"
            with open(temp_path, "wb") as f:
                f.write(img_data)
            image_path = temp_path
        elif request.image_path:
            if not os.path.exists(request.image_path):
                raise HTTPException(status_code=400, detail=f"Image not found: {request.image_path}")
            image_path = request.image_path
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        avatar_id, source_info = _register_avatar(sdk, image_path, request.avatar_id)

        if request.prerender_clips:
            _start_prerender(avatar_id, image_path)

        return {
            "status": "registered",
            "avatar_id": avatar_id,
            "inference_ready": True,
            "clips_ready": _clips_ready(avatar_id),
            "prerender_status": prerender_jobs.get(avatar_id, {}).get("status", "none"),
            "size_mb": round(_estimate_size_mb(source_info), 1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Key conventions:**
- Route handlers are `async def`
- Pydantic `BaseModel` for request bodies (see [Section 10](#10-api-response-format) for model definitions)
- Input decoded/validated at top of handler
- Business logic delegated to private `_functions()`
- Success returns a plain dict (FastAPI auto-serializes to JSON)
- Top-level `except Exception` catches everything and returns HTTP 500

---

## 2. Data Access Pattern

There is **no SQL database**. All persistence uses Python's `pickle` for disk caching and in-memory `dict` for runtime state. This is the canonical read/write pattern.

**Source:** `ditto/ditto_api.py` (lines ~166–182)

```python
def _cache_path(avatar_id: str) -> Path:
    return CACHE_DIR / f"{avatar_id}.pkl"


def _save_cache(avatar_id: str, source_info: dict):
    """Persist source_info to disk."""
    with open(_cache_path(avatar_id), "wb") as f:
        pickle.dump(source_info, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(avatar_id: str) -> dict | None:
    """Load source_info from disk."""
    path = _cache_path(avatar_id)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
```

**Runtime state uses module-level dictionaries:**

```python
# In-memory stores (ditto/ditto_api.py, lines ~113–121)
avatar_cache: dict[str, dict] = {}          # avatar_id → source_info features
active_sessions: dict[str, dict] = {}       # session_id → LiveKit session state
prerender_jobs: dict[str, dict] = {}        # avatar_id → {status, started, error}
```

**Key conventions:**
- Pickle for heavy serialized data (9MB+ feature dicts), not JSON
- `pickle.HIGHEST_PROTOCOL` for best performance
- `_load_cache()` returns `None` on miss — caller checks
- Path construction via `pathlib.Path`, not string concatenation
- Directories created at module load: `CACHE_DIR.mkdir(parents=True, exist_ok=True)`

---

## 3. Error Handling

Errors follow a two-tier pattern: **validation errors** raise `HTTPException` with 400, **internal failures** are caught with a blanket try/except returning 500.

**Source:** `ditto/ditto_api.py` (lines ~1105–1137)

```python
@app.post("/generate")
async def generate_video(request: GenerateRequest):
    """Generate video from audio using a cached avatar."""
    sdk = load_ditto()

    # Tier 1: Validation errors → 400
    if request.avatar_id not in avatar_cache:
        raise HTTPException(status_code=400, detail=f"Avatar '{request.avatar_id}' not found. Call /register first.")

    try:
        if request.audio_base64:
            audio_data = base64.b64decode(request.audio_base64)
            audio_path = "/tmp/audio_temp.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_data)
        elif request.audio_path:
            audio_path = request.audio_path
        else:
            raise HTTPException(status_code=400, detail="No audio provided")

        video_data = _generate_video(
            sdk, request.avatar_id, audio_path,
            sampling_timesteps=request.sampling_timesteps,
            fps=request.fps,
        )

        return Response(content=video_data, media_type="video/mp4")

    # Tier 2: Internal errors → 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**SmolVLM alternative pattern** — `JSONResponse` instead of `HTTPException`:

**Source:** `smolvlm/smolvlm_server.py` (lines ~110–120)

```python
try:
    # ... inference logic ...
    return JSONResponse({
        "success": True,
        "response": generated_text.strip(),
        "prompt": request.prompt
    })
except Exception as e:
    return JSONResponse({"success": False, "error": str(e)}, status_code=500)
```

**Key conventions:**
- `HTTPException(status_code=400)` for client errors (missing input, unknown ID)
- `HTTPException(status_code=500)` for server errors (model failure, I/O error)
- Error messages are human-readable strings, not error codes
- WebSocket errors sent as JSON: `{"type": "error", "message": "..."}`
- No custom exception classes — plain `HTTPException` everywhere

---

## 4. Authentication & Middleware

**There is no authentication.** The only middleware is CORS, applied identically across all services. This section documents what exists, not what should exist.

**Source:** `chatterbox/api_server.py` (lines ~26–34)

```python
app = FastAPI(title="Chatterbox TTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Compact variant** — `ditto/ditto_api.py` (lines ~54–55):

```python
app = FastAPI(title="Ditto Avatar API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**The closest thing to auth** — LiveKit agent identity check in `ditto/ditto_api.py`:

```python
# Inside _audio_receiver_task (line ~1501)
def _on_byte_stream(reader: lk_rtc.ByteStreamReader, participant_id: str):
    if participant_id != agent_identity:
        return  # Ignore audio from unrecognized participants
    reader_queue.put_nowait(reader)

# Inside RPC handler (line ~1505)
def _on_clear_buffer(data: lk_rtc.RpcInvocationData) -> str:
    if data.caller_identity != agent_identity:
        return "reject"
    # ... handle interrupt ...
    return "ok"
```

**Key conventions:**
- No API keys, Bearer tokens, JWT validation, or session cookies in any service
- All services bind to `0.0.0.0` — assumed to run behind a reverse proxy or on a private network
- LiveKit JWT tokens are passed by the client and validated by the LiveKit server, not this codebase
- Agent identity is checked for LiveKit DataStream/RPC, but this is participant filtering, not authentication

---

## 5. Environment Variables & Config

All config uses `os.environ.get()` with sensible defaults. No `.env` files, no config libraries, no settings classes.

**Source:** `ditto/ditto_api.py` (lines ~33, 63–79, 125–127)

```python
# Path configuration — all use Path() for cross-platform support
DITTO_PATH = Path(os.environ.get("DITTO_PATH", str(Path(__file__).resolve().parent / "ditto-talkinghead")))

CACHE_DIR = Path(os.environ.get("AVATAR_CACHE_DIR", "/workspace/avatar_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLIPS_DIR = Path(os.environ.get("AVATAR_CLIPS_DIR", "/workspace/avatar_clips"))
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = Path(os.environ.get("AVATAR_IMAGES_DIR", "/workspace/avatar_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# URL configuration
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8282/tts/stream")

# Backend selection (string enum)
backend = os.environ.get("DITTO_BACKEND", "onnx")

# Nested derived path
SKYREELS_PATH = Path(os.environ.get("SKYREELS_PATH", "/workspace/SkyReels-A1"))
SKYREELS_PYTHON = Path(os.environ.get("SKYREELS_PYTHON", str(SKYREELS_PATH / ".venv" / "bin" / "python")))
```

**Key conventions:**
- `os.environ.get("NAME", default)` — always provide a default
- Directories auto-created with `mkdir(parents=True, exist_ok=True)` at module load
- `Path()` wrapping for all filesystem paths
- `ALL_CAPS` naming for module-level config constants
- No validation or type coercion — values used as-is (string or Path)

---

## 6. Module / Feature Organization

Pipeline components follow a class-based pattern: a top-level wrapper class that delegates to a specialized implementation, with constants and configuration at the module level.

**Source:** `ditto/ditto-talkinghead/core/atomic_components/wav2feat.py`

```python
"""Audio Waveform to Feature Extraction: converts raw audio into HuBERT features.

HuBERT (Hidden-Unit BERT) produces 1024-dim features at 50Hz (one feature per 20ms).
These features encode phonetic content that drives the LMDM motion model.
"""

import math
import numpy as np
import librosa

from ..aux_models.hubert_stream import HubertStreaming

# Module-level constants
HUBERT_FEATURE_DIM = 1024
HUBERT_HOP_SAMPLES = 320   # 16000 Hz / 50 features/sec
TARGET_SAMPLE_RATE = 16000
TARGET_FPS = 25


class Wav2Feat:
    """Unified audio-to-feature interface supporting HuBERT extraction."""

    def __init__(self, w2f_cfg: dict, w2f_type: str = "hubert"):
        self.feature_type = w2f_type.lower()
        if self.feature_type == "hubert":
            self.extractor = Wav2FeatHubert(hubert_cfg=w2f_cfg)
            self.feat_dim = HUBERT_FEATURE_DIM
            self.support_streaming = True
        else:
            raise ValueError(f"Unsupported feature type: {w2f_type}")

    def __call__(self, audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE, ...) -> np.ndarray:
        """Extract features from audio."""
        if self.feature_type == "hubert":
            return self.extractor(audio, chunksize=chunksize)
        raise ValueError(f"Unsupported feature type: {self.feature_type}")
```

**Component inter-dependency** — `ditto/ditto-talkinghead/core/atomic_components/avatar_registrar.py`:

```python
import numpy as np

from .loader import load_source_frames
from .source2info import Source2Info
```

**Key conventions:**
- One class per file, file named after the component
- Module docstring explains what the component does and its role in the pipeline
- `__init__` takes a config dict (`_cfg` suffix) and a type selector string
- `__call__` makes the class callable (pipeline stage pattern)
- Relative imports (`from ..aux_models`) for internal dependencies
- Constants at module level, `ALL_CAPS`
- Config dicts passed top-down, not read from env inside components

---

## 7. Test Patterns

There is **no formal test framework** (no pytest, unittest, or CI). The only test-like file is a manual CLI script.

**Source:** `skyreels/skyreels_a1/src/lmk3d_test.py`

```python
class FaceAnimationProcessor:
    def __init__(self, device='cuda', checkpoint="pretrained_models/smirk/smirk_encoder.pt"):
        self.device = device
        self.app = FaceAnalysis(allowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.smirk_encoder = SmirkEncoder().to(device)
        self.flame = FLAME(n_shape=300, n_exp=50).to(device)
        self.renderer = Renderer().to(device)
        self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        checkpoint_data = torch.load(checkpoint)
        checkpoint_encoder = {
            k.replace('smirk_encoder.', ''): v
            for k, v in checkpoint_data.items() if 'smirk_encoder' in k
        }
        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_encoder.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and image for face animation.")
    parser.add_argument('--source_image', type=str, default="assets/ref_images/1.png")
    parser.add_argument('--driving_video', type=str, default="assets/driving_video/1.mp4")
    parser.add_argument('--output_path', type=str, default="./output.mp4")
    args = parser.parse_args()

    processor = FaceAnimationProcessor(checkpoint='./pretrained_models/smirk/SMIRK_em1.pt')
    processor.process_video(args.source_image, args.driving_video, args.output_path)
```

**Other "test" mechanisms:**
- `ditto/test_viewer.html` — HTML page that connects to `WS /ws/test_frames` for browser-based debugging
- `WS /ws/test_frames` endpoint in `ditto/ditto_api.py` — generates sine-wave audio and streams JPEG frames

**Key conventions (such as they are):**
- Manual CLI scripts with `argparse`, run via `python script.py --args`
- `if __name__ == "__main__":` guard for script entry
- No assertions, no test runners, no expected-output validation
- Test files named `*_test.py` (not `test_*.py` pytest convention)

---

## 8. Background Jobs / Queue Tasks

Background work uses `ThreadPoolExecutor` for CPU/subprocess tasks and `asyncio.create_task` for async I/O tasks. Status tracked via module-level dictionaries.

**Source:** `ditto/ditto_api.py` (lines ~120–123, 547–606)

### Executor Setup

```python
prerender_jobs: dict[str, dict] = {}
prerender_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prerender")
```

### Job Submission

```python
def _start_prerender(avatar_id: str, image_path: str):
    """Submit a pre-render job to the background executor."""
    if avatar_id in prerender_jobs and prerender_jobs[avatar_id]["status"] in ("running", "done"):
        return  # already running or finished

    persistent_path = _avatar_image_path(avatar_id)
    if not persistent_path.exists():
        shutil.copy2(image_path, persistent_path)

    prerender_jobs[avatar_id] = {"status": "pending"}
    prerender_executor.submit(_run_prerender, avatar_id, str(persistent_path))
```

### Job Worker

```python
def _run_prerender(avatar_id: str, image_path: str):
    """Run SkyReels-A1 pre-render in a background thread."""
    try:
        if not _ensure_skyreels_setup(avatar_id):
            return

        prerender_jobs[avatar_id] = {"status": "running", "started": time.time()}

        cmd = [
            str(SKYREELS_PYTHON), str(PRERENDER_SCRIPT),
            "--image_path", image_path,
            "--output_dir", str(CLIPS_DIR),
            "--avatar_id", avatar_id,
            "--clips", *CLIP_TYPES,
            "--target_fps", "24",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SKYREELS_PATH)
        result = subprocess.run(cmd, cwd=str(SKYREELS_PATH), env=env,
                                capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            prerender_jobs[avatar_id] = {"status": "done", "finished": time.time()}
        else:
            prerender_jobs[avatar_id] = {
                "status": "failed",
                "error": result.stderr[-500:] if result.stderr else "unknown error",
            }

    except subprocess.TimeoutExpired:
        prerender_jobs[avatar_id] = {"status": "failed", "error": "timeout (30min)"}
    except Exception as e:
        prerender_jobs[avatar_id] = {"status": "failed", "error": str(e)}
```

### Async Task Pattern (LiveKit sessions)

```python
# In POST /start_session handler (lines ~1804–1816)
publisher_task = asyncio.create_task(
    _unified_publisher(session_id, active_sessions[session_id])
)
audio_recv_task = asyncio.create_task(
    _audio_receiver_task(session_id, active_sessions[session_id], request.agent_identity)
)

# Cleanup in _auto_stop_session()
session["audio_recv_task"].cancel()
session["publisher_task"].cancel()
await session["av_sync"].aclose()
await session["room"].disconnect()
```

**Key conventions:**
- `ThreadPoolExecutor` for subprocess/blocking work (single worker to serialize GPU access)
- `asyncio.create_task` for concurrent async work within the event loop
- Status tracked in a plain dict: `{"status": "pending"|"running"|"done"|"failed", "error": ...}`
- Idempotency: check if job already running/done before submitting
- Timeouts on all subprocesses (`timeout=1800` = 30min)
- Tasks cancelled and awaited during cleanup

---

## 9. Frontend Components

**There is no frontend application.** No React, Vue, Angular, Svelte, or similar framework exists in this repository.

The only client-side code is two standalone HTML debug pages:

**Source:** `ditto/test_viewer.html`

```html
<!-- Minimal HTML page that connects to WS /ws/test_frames -->
<!-- Opens WebSocket, receives base64 JPEG frames, renders to <img> tag -->
<!-- No build step, no bundler, no framework — plain HTML + vanilla JS -->
```

**Source:** `skyreels/test_viewer.html`

```html
<!-- Same pattern — standalone debug viewer, no framework -->
```

These are development utilities, not a user-facing frontend. The actual client interface is expected to be a separate application that connects via LiveKit WebRTC.

---

## 10. API Response Format

### Pydantic Request Models

All request bodies use `pydantic.BaseModel` with `Optional` fields and defaults.

**Source:** `ditto/ditto_api.py` (lines ~130–157)

```python
class RegisterAvatarRequest(BaseModel):
    avatar_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    prerender_clips: bool = True


class GenerateRequest(BaseModel):
    avatar_id: str
    audio_base64: Optional[str] = None
    audio_path: Optional[str] = None
    sampling_timesteps: Optional[int] = None
    fps: Optional[int] = None


class TextGenerateRequest(BaseModel):
    avatar_id: str
    text: str
    voice: str = "tara"


class StartSessionRequest(BaseModel):
    avatar_id: str
    livekit_url: str
    livekit_token: str
    fps: int = DEFAULT_FPS
    sampling_timesteps: int = DEFAULT_SAMPLING_TIMESTEPS
    agent_identity: str | None = None
```

**Source:** `smolvlm/smolvlm_server.py` (lines ~21–25)

```python
class ImageQueryRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    prompt: str = "Describe this image in detail."
```

### Success Response Shapes

**Pattern A — Plain dict** (most endpoints):

```python
# ditto/ditto_api.py — POST /register
return {
    "status": "registered",
    "avatar_id": avatar_id,
    "inference_ready": True,
    "clips_ready": _clips_ready(avatar_id),
    "prerender_status": prerender_jobs.get(avatar_id, {}).get("status", "none"),
    "size_mb": round(_estimate_size_mb(source_info), 1),
}
```

**Pattern B — Collection response** (list endpoints):

```python
# ditto/ditto_api.py — GET /avatars
return {"avatars": [...], "total": len(avatars)}
```

**Pattern C — Binary response** (file endpoints):

```python
# ditto/ditto_api.py — POST /generate
return Response(content=video_data, media_type="video/mp4")

# chatterbox/api_server.py — POST /api/tts
return Response(
    content=buffer.read(),
    media_type="audio/wav",
    headers={"Content-Disposition": f"attachment; filename=tts_{request_id}.wav"}
)
```

**Pattern D — JSONResponse with success flag** (SmolVLM):

```python
# smolvlm/smolvlm_server.py — POST /v1/query
return JSONResponse({
    "success": True,
    "response": generated_text.strip(),
    "prompt": request.prompt
})
```

### Error Response Shapes

**Pattern A — HTTPException** (Ditto, Chatterbox):

```python
# Validation error (400)
raise HTTPException(status_code=400, detail=f"Avatar '{avatar_id}' not found. Call /register first.")

# Server error (500)
raise HTTPException(status_code=500, detail=str(e))

# Produces: {"detail": "error message"}
```

**Pattern B — JSONResponse error** (SmolVLM):

```python
return JSONResponse({"success": False, "error": str(e)}, status_code=500)
```

**Pattern C — WebSocket error** (Chatterbox):

```python
await websocket.send_json({"type": "error", "message": "Voice prompt not initialized"})
```

**Note:** There is no unified error envelope across services. Ditto and Chatterbox use `{"detail": "..."}` (FastAPI default), SmolVLM uses `{"success": false, "error": "..."}`, WebSockets use `{"type": "error", "message": "..."}`.

---

## 11. Naming Conventions

### Files

| Convention | Examples |
|---|---|
| **Snake_case** for Python files | `ditto_api.py`, `api_server.py`, `smolvlm_server.py`, `prerender_clips.py` |
| **Snake_case** for component files | `audio2motion.py`, `avatar_registrar.py`, `wav2feat.py`, `motion_stitch.py` |
| **Kebab-case** for directories with dashes | `ditto-talkinghead/` |
| **Snake_case** for directories without dashes | `atomic_components/`, `aux_models/`, `media_pipe/` |
| **ALL_CAPS** for model/config directories | `FLAME/` |
| **Test files** use `_test.py` suffix | `lmk3d_test.py` (not `test_lmk3d.py`) |

### Functions

| Convention | Examples |
|---|---|
| **snake_case** for all functions | `register_avatar()`, `generate_video()`, `list_avatars()` |
| **Leading underscore** for private/internal | `_register_avatar()`, `_save_cache()`, `_load_cache()`, `_warm_load_all()`, `_clips_ready()` |
| **No prefix** for public route handlers | `register_avatar()`, `generate_video()`, `health()` |
| **Verb-first** for actions | `load_ditto()`, `get_model()`, `_start_prerender()`, `_run_prerender()` |

### Variables

| Convention | Examples |
|---|---|
| **snake_case** for locals and instance vars | `avatar_cache`, `source_info`, `audio_16k`, `video_data`, `frame_count` |
| **ALL_CAPS** for module constants | `DITTO_PATH`, `CACHE_DIR`, `CLIPS_DIR`, `DEFAULT_FPS`, `AUDIO_DELAY_FRAMES`, `CLIP_TYPES`, `HUBERT_FEATURE_DIM` |
| **_cfg suffix** for config dicts | `w2f_cfg`, `hubert_cfg`, `lmdm_cfg` |
| **_lst suffix** for lists of per-frame data | `x_s_info_lst`, `img_rgb_lst`, `blink_lst`, `eye_open_n_lst` |
| **_path suffix** for file paths | `image_path`, `audio_path`, `cache_path` |

### Classes

| Convention | Examples |
|---|---|
| **PascalCase** for all classes | `Wav2Feat`, `Audio2Motion`, `AvatarRegistrar`, `Source2Info` |
| **PascalCase** for Pydantic models | `RegisterAvatarRequest`, `GenerateRequest`, `StartSessionRequest`, `ImageQueryRequest` |
| **PascalCase** with abbreviations | `TRTWrapper`, `AVSynchronizer`, `HubertStreaming` |

### API Paths

| Convention | Examples |
|---|---|
| **Lowercase, no trailing slash** | `/register`, `/generate`, `/health` |
| **Underscore-separated** for multi-word | `/generate_from_text`, `/generate_frames` |
| **Resource-based nesting** | `/avatars`, `/avatars/{avatar_id}`, `/avatars/{avatar_id}/status`, `/avatars/{avatar_id}/resolution` |
| **Action as sub-path** | `/avatars/{avatar_id}/prerender`, `/stop_session/{session_id}` |
| **Versioned prefix** (SmolVLM only) | `/v1/query`, `/v1/query/upload`, `/v1/unload` |

### Constants and Enums

Enums are not used. String literals are compared directly:

```python
backend = os.environ.get("DITTO_BACKEND", "onnx")  # "trt" or "onnx"
```

```python
CLIP_TYPES = ["idle", "thinking", "lookingup", "lippurse", "greeting"]
```

---

## 12. Import Patterns

### Standard Import Order

Imports follow: **stdlib → sys.path manipulation → third-party → internal**. No import sorting tools are configured.

**Source:** `ditto/ditto_api.py` (lines ~1–46)

```python
# 1. Standard library
import sys
import io
import os
import time
import base64
import math
import pickle
import hashlib
import shutil
import asyncio
import random
import threading
import subprocess
import struct
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# 2. sys.path manipulation (before internal imports)
DITTO_PATH = Path(os.environ.get("DITTO_PATH", ...))
sys.path.insert(0, str(DITTO_PATH))

# 3. Third-party
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse, Response, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import librosa
import httpx
import soundfile as sf
from scipy.signal import resample_poly

# 4. Optional imports with fallback
try:
    from livekit import rtc as lk_rtc, api as lk_api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
```

### Internal Component Imports

Components use **relative imports** within their package.

**Source:** `ditto/ditto-talkinghead/core/atomic_components/wav2feat.py`

```python
import math
import numpy as np
import librosa

from ..aux_models.hubert_stream import HubertStreaming
```

**Source:** `ditto/ditto-talkinghead/core/atomic_components/avatar_registrar.py`

```python
import numpy as np

from .loader import load_source_frames
from .source2info import Source2Info
```

### Key Conventions

| Pattern | Usage |
|---|---|
| **Relative imports** (`from ..module`) | Within `ditto-talkinghead/core/` package hierarchy |
| **sys.path.insert** | API servers adding engine paths before importing engine modules |
| **try/except ImportError** | Optional dependencies (LiveKit) with `_AVAILABLE` boolean flag |
| **`import X as Y`** | Used for LiveKit: `from livekit import rtc as lk_rtc` |
| **No path aliases** | No `__init__.py` re-exports, no `__all__`, no barrel files |
| **No default exports** | Python doesn't have them — all imports are named |
| **numpy always as np** | `import numpy as np` (universal convention) |
| **cv2 bare import** | `import cv2` (not aliased) |

---

## Quick Reference Summary

| Aspect | Convention |
|---|---|
| **Web framework** | FastAPI + async handlers |
| **Request validation** | Pydantic `BaseModel` with `Optional` fields |
| **Response format** | Plain dict (auto-JSON), `Response()` for binary, `JSONResponse()` for explicit |
| **Error format** | `HTTPException(status_code, detail=str)` |
| **Data persistence** | `pickle` files + in-memory dicts |
| **Config** | `os.environ.get("NAME", default)` at module level |
| **Background work** | `ThreadPoolExecutor` for blocking, `asyncio.create_task` for async |
| **File naming** | `snake_case.py` |
| **Function naming** | `snake_case`, `_private_prefix` |
| **Class naming** | `PascalCase` |
| **Constant naming** | `ALL_CAPS` |
| **API path naming** | `/lowercase_underscored`, resource nesting with `{id}` |
| **Import order** | stdlib → path setup → third-party → (try optional) → internal |
| **Internal imports** | Relative (`from ..module import Class`) |
| **Auth** | None (assumed behind reverse proxy) |
| **Tests** | None (manual CLI scripts only) |
| **Frontend** | None (standalone HTML debug pages only) |
