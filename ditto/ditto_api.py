"""Ditto Avatar Streaming API Server - Port 8181

Real-time talking head generation from audio with avatar caching.

Lifecycle for a new avatar:
  1. POST /register  → Ditto registration (~2-3s), avatar immediately usable for inference
  2. Background job  → SkyReels-A1 pre-renders idle/thinking/greeting clips (minutes, non-blocking)
  3. GET /avatars    → Shows readiness: inference_ready=true, clips_ready=true/false
  4. GET /clips/{id}/{type} → Serve pre-rendered clip once available

Caches pre-computed avatar features (~9MB each) to skip registration on repeat calls.
"""

import sys
import io
import os
import time
import base64
import math
import logging
import pickle
import hashlib
import shutil
import asyncio
import random
import threading
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("ditto")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

# Add Ditto to path
DITTO_PATH = Path(os.environ.get("DITTO_PATH", str(Path(__file__).resolve().parent / "ditto-talkinghead")))
sys.path.insert(0, str(DITTO_PATH))

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse, Response, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import cv2
import librosa
import httpx
import struct
import soundfile as sf
from scipy.signal import resample_poly

try:
    from livekit import rtc as lk_rtc, api as lk_api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

app = FastAPI(title="Ditto Avatar API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Simple in-memory rate limiter (per-IP, token bucket)
_rate_limit_buckets: dict[str, list] = {}  # ip -> [timestamp, ...]
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "60"))  # requests per minute


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_limit_buckets.setdefault(client_ip, [])
    # Purge entries older than 60s
    bucket[:] = [t for t in bucket if now - t < 60]
    if len(bucket) >= RATE_LIMIT_RPM:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
    bucket.append(now)
    return await call_next(request)


# Global instances
ditto_sdk = None
avatar_cache: dict[str, dict] = {}  # avatar_id -> source_info
avatar_cache_lock = threading.Lock()
active_sessions: dict[str, dict] = {}  # session_id -> {room, video_source, output_queue, ...}
active_sessions_lock = threading.Lock()


CACHE_DIR = Path(os.environ.get("AVATAR_CACHE_DIR", "/workspace/avatar_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLIPS_DIR = Path(os.environ.get("AVATAR_CLIPS_DIR", "/workspace/avatar_clips"))
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = Path(os.environ.get("AVATAR_IMAGES_DIR", "/workspace/avatar_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Pipeline Defaults ─────────────────────────────────────────────
# DDIM diffusion steps for lip-sync generation. 5 steps balances quality/speed
# at ~25-30 iterations/sec in streaming mode (vs 50 steps offline default).
DEFAULT_SAMPLING_TIMESTEPS = 5
# Video frame rate. 25fps matches the Ditto training data and LiveKit VP8 output.
DEFAULT_FPS = 25
# Audio is delayed 5 frames (200ms at 25fps) relative to video to hide the
# LMDM diffusion model's initial ramp-up latency on the first few frames.
AUDIO_DELAY_FRAMES = 5

# ─── TTS Integration ──────────────────────────────────────────────
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8282/tts/stream")
_tts_http_client: httpx.Client | None = None


def _get_tts_client() -> httpx.Client:
    """Get or create a reusable httpx client for TTS requests (connection pooling)."""
    global _tts_http_client
    if _tts_http_client is None or _tts_http_client.is_closed:
        _tts_http_client = httpx.Client(timeout=120.0, limits=httpx.Limits(max_keepalive_connections=5))
    return _tts_http_client
TTS_SAMPLE_RATE = 24000     # Chatterbox TTS outputs 24kHz audio
DITTO_SAMPLE_RATE = 16000   # Ditto's HuBERT expects 16kHz input
CHUNK_SAMPLES_24K = 9720    # 0.405s chunk at 24kHz (matches streaming granularity)
CHUNK_SAMPLES_16K = 6480    # 0.405s chunk at 16kHz (24kHz resampled by 2/3)
WAV_HEADER_SIZE = 44        # Standard WAV header size to skip in streaming TTS

# Larger chunk processing for better HuBERT feature rate.
# Standard chunksize=(3,5,2) yields only 5 features per 6480-sample chunk (12.3 feat/s).
# Larger chunksize=(3,35,2) yields 35 features per 25680-sample chunk (21.8 feat/s).
# The LMDM model needs 70 features per window, so larger chunks fill windows ~2x faster.
STANDARD_CHUNKSIZE = (3, 5, 2)
STANDARD_CHUNK_SAMPLES = CHUNK_SAMPLES_16K  # 6480
STANDARD_CHUNK_BYTES = STANDARD_CHUNK_SAMPLES * 2  # 12960



# LiveKit DataStream constants (matching livekit-agents avatar protocol)
AUDIO_STREAM_TOPIC = "lk.audio_stream"
RPC_CLEAR_BUFFER = "lk.clear_buffer"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"


def _get_chunk_config() -> tuple:
    """Return (chunksize, chunk_samples, chunk_bytes) based on backend.

    TRT hubert engine supports input sizes up to 12960 samples.
    Medium chunks (3,15,2) = 12880 samples → 15 features per call.
    ONNX/PyTorch can use even larger chunks.
    """
    # Use STANDARD chunksize (3,5,2) to match offline pipeline's wav2feat.wav2feat()
    # which also uses (3,5,2) by default. Different chunksizes produce different
    # HuBERT features due to transformer self-attention context differences.
    backend = os.environ.get("DITTO_BACKEND", "onnx")
    STANDARD_CHUNK_SAMPLES = int(sum(STANDARD_CHUNKSIZE) * 0.04 * 16000) + 80  # 6480
    STANDARD_CHUNK_BYTES = STANDARD_CHUNK_SAMPLES * 2  # 12960
    return STANDARD_CHUNKSIZE, STANDARD_CHUNK_SAMPLES, STANDARD_CHUNK_BYTES

# Pre-render clip types
CLIP_TYPES = ["idle", "thinking", "lookingup", "lippurse", "greeting"]

# Background pre-render state: avatar_id -> {"status": "pending"|"running"|"done"|"failed", "error": ...}
prerender_jobs: dict[str, dict] = {}
prerender_jobs_lock = threading.Lock()
prerender_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prerender")

# Path to prerender script (SkyReels-A1) — uses its own venv since it needs different deps
SKYREELS_PATH = Path(os.environ.get("SKYREELS_PATH", "/workspace/SkyReels-A1"))
SKYREELS_PYTHON = Path(os.environ.get("SKYREELS_PYTHON", str(SKYREELS_PATH / ".venv" / "bin" / "python")))
PRERENDER_SCRIPT = Path(os.environ.get("PRERENDER_SCRIPT", str(Path(__file__).resolve().parent / "prerender_clips.py")))


class RegisterAvatarRequest(BaseModel):
    avatar_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    prerender_clips: bool = True  # whether to kick off background clip generation


class GenerateRequest(BaseModel):
    avatar_id: str
    audio_base64: Optional[str] = None
    audio_path: Optional[str] = None
    sampling_timesteps: Optional[int] = Field(None, ge=1, le=50)
    fps: Optional[int] = Field(None, ge=1, le=60)


class TextGenerateRequest(BaseModel):
    avatar_id: str
    text: str
    voice: str = "tara"


class StartSessionRequest(BaseModel):
    avatar_id: str
    livekit_url: str        # wss://lk.scoreexl.com
    livekit_token: str      # JWT for avatar participant
    fps: int = Field(DEFAULT_FPS, ge=1, le=60)
    sampling_timesteps: int = Field(DEFAULT_SAMPLING_TIMESTEPS, ge=1, le=50)
    agent_identity: str | None = None  # Identity of the agent sending audio via DataStream


MAX_BASE64_SIZE = 50 * 1024 * 1024  # 50MB max for base64 uploads

# Allowed parent directories for user-supplied file paths
_ALLOWED_PATH_ROOTS = [CACHE_DIR, CLIPS_DIR, IMAGES_DIR, Path("/tmp"), Path("/workspace")]


def _validate_file_path(user_path: str) -> Path:
    """Validate a user-supplied file path against path traversal attacks.
    Raises HTTPException if path is outside allowed directories."""
    resolved = Path(user_path).resolve()
    for allowed in _ALLOWED_PATH_ROOTS:
        try:
            resolved.relative_to(allowed.resolve())
            return resolved
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail=f"Path not allowed: {user_path}")


def _to_mono_int16(pcm: np.ndarray, num_channels: int) -> np.ndarray:
    """Convert interleaved PCM int16 to mono int16."""
    if num_channels <= 1:
        return pcm
    usable = (len(pcm) // num_channels) * num_channels
    if usable <= 0:
        return np.zeros((0,), dtype=np.int16)
    pcm = pcm[:usable].reshape(-1, num_channels).astype(np.int32)
    return np.mean(pcm, axis=1).clip(-32768, 32767).astype(np.int16)


def _resample_int16(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample PCM int16 using polyphase filtering."""
    if src_sr == dst_sr or len(pcm) == 0:
        return pcm
    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    out = resample_poly(pcm.astype(np.float64), up, down)
    return np.round(out).clip(-32768, 32767).astype(np.int16)


def _image_hash(image_path: str) -> str:
    """Generate a stable avatar_id from image contents."""
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _cache_path(avatar_id: str) -> Path:
    return CACHE_DIR / f"{avatar_id}.pkl"


import hmac as _hmac

# Secret for HMAC pickle validation — prevents loading tampered cache files.
# In production, set PICKLE_HMAC_SECRET env var to a random value.
_PICKLE_SECRET = os.environ.get("PICKLE_HMAC_SECRET", "ditto-avatar-cache-default-key").encode()


def _save_cache(avatar_id: str, source_info: dict):
    """Persist source_info to disk atomically with HMAC integrity check."""
    final_path = _cache_path(avatar_id)
    tmp_path = final_path.with_suffix(".pkl.tmp")
    data = pickle.dumps(source_info, protocol=pickle.HIGHEST_PROTOCOL)
    sig = _hmac.new(_PICKLE_SECRET, data, "sha256").digest()
    with open(tmp_path, "wb") as f:
        f.write(sig)  # 32-byte HMAC prefix
        f.write(data)
    tmp_path.replace(final_path)


def _load_cache(avatar_id: str) -> dict | None:
    """Load source_info from disk with HMAC integrity verification."""
    path = _cache_path(avatar_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        content = f.read()
    if len(content) < 32:
        logger.info(f"[cache] WARNING: {path} too small, skipping")
        return None
    stored_sig = content[:32]
    data = content[32:]
    expected_sig = _hmac.new(_PICKLE_SECRET, data, "sha256").digest()
    if not _hmac.compare_digest(stored_sig, expected_sig):
        # Legacy file without HMAC — load but re-save with HMAC
        logger.info(f"[cache] {path.name}: no valid HMAC (legacy file), loading and re-signing...")
        try:
            with open(path, "rb") as f:
                result = pickle.load(f)
            _save_cache(avatar_id, result)  # re-save with HMAC
            return result
        except Exception as e:
            logger.info(f"[cache] WARNING: Failed to load legacy cache {path}: {e}")
            return None
    return pickle.loads(data)


def _warm_load_all():
    """Load all cached avatars from disk on startup."""
    global avatar_cache
    for pkl_file in CACHE_DIR.glob("*.pkl"):
        avatar_id = pkl_file.stem
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            with avatar_cache_lock:
                avatar_cache[avatar_id] = data
            logger.info(f"  Loaded cached avatar: {avatar_id}")
        except Exception as e:
            logger.info(f"  Failed to load cache {pkl_file}: {e}")
    logger.info(f"Loaded {len(avatar_cache)} cached avatars from disk")


def _resolve_backend(backend: str) -> tuple[str, str]:
    """Resolve backend name to (config_pkl, data_dir) based on available checkpoints."""
    ckpt = DITTO_PATH / "checkpoints"
    if backend == "trt":
        # Prefer v10 engines (custom-built for exact TRT version), fall back to Ampere+
        if (ckpt / "ditto_trt_v10").exists() and (ckpt / "ditto_cfg" / "v0.4_hubert_cfg_trt_v10.pkl").exists():
            return "v0.4_hubert_cfg_trt_v10.pkl", "ditto_trt_v10"
        elif (ckpt / "ditto_trt_Ampere_Plus").exists():
            # Prefer full TRT config (warp_network_fp16.engine) if engine exists
            if (ckpt / "ditto_trt_Ampere_Plus" / "warp_network_fp16.engine").exists() and (ckpt / "ditto_cfg" / "v0.4_hubert_cfg_trt.pkl").exists():
                return "v0.4_hubert_cfg_trt.pkl", "ditto_trt_Ampere_Plus"
            # Fall back to v10 config (warp_network.pth) if no TRT warp engine
            elif (ckpt / "ditto_cfg" / "v0.4_hubert_cfg_trt_v10.pkl").exists():
                return "v0.4_hubert_cfg_trt_v10.pkl", "ditto_trt_Ampere_Plus"
        else:
            logger.warning(f" DITTO_BACKEND=trt but no TRT engines found, falling back to pytorch")
            return "v0.4_hubert_cfg_pytorch.pkl", "ditto_pytorch"
    else:
        return "v0.4_hubert_cfg_pytorch.pkl", "ditto_pytorch"



# ─── Monkey-patch: setup_from_cache for OFFLINE StreamSDK ───────────────────

def _setup_from_cache(self, source_info, output_path, **kwargs):
    """Like setup() but uses pre-cached source_info instead of running avatar_registrar.

    This skips the expensive registration (4 neural-net forward passes) and reuses
    the cached tensors from a prior call to avatar_registrar.register().
    """
    import threading
    import queue
    from tqdm import tqdm
    from core.atomic_components.writer import VideoWriterByImageIO
    from core.atomic_components.cfg import print_cfg

    # ======== Prepare Options ========
    kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
    logger.info('=' * 20 + ' setup_from_cache kwargs ' + '=' * 20)
    print_cfg(**kwargs)
    logger.info('=' * 50)

    # -- avatar_registrar: template cfg --
    self.max_size = kwargs.get('max_size', 1920)
    self.template_n_frames = kwargs.get('template_n_frames', -1)

    # -- avatar_registrar: crop cfg --
    self.crop_scale = kwargs.get('crop_scale', 2.3)
    self.crop_vx_ratio = kwargs.get('crop_vx_ratio', 0)
    self.crop_vy_ratio = kwargs.get('crop_vy_ratio', -0.125)
    self.crop_flag_do_rot = kwargs.get('crop_flag_do_rot', True)

    # -- avatar_registrar: smo for video --
    self.smo_k_s = kwargs.get('smo_k_s', 13)

    # -- condition_handler: ECS --
    self.emo = kwargs.get('emo', 4)
    self.eye_f0_mode = kwargs.get('eye_f0_mode', False)
    self.ch_info = kwargs.get('ch_info', None)

    # -- audio2motion: setup --
    self.overlap_v2 = kwargs.get('overlap_v2', 10)
    self.fix_kp_cond = kwargs.get('fix_kp_cond', 0)
    self.fix_kp_cond_dim = kwargs.get('fix_kp_cond_dim', None)
    self.sampling_timesteps = kwargs.get('sampling_timesteps', 50)
    self.online_mode = kwargs.get('online_mode', False)
    self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
    self.smo_k_d = kwargs.get('smo_k_d', 3)

    # -- motion_stitch: setup --
    self.N_d = kwargs.get('N_d', -1)
    self.use_d_keys = kwargs.get('use_d_keys', None)
    self.relative_d = kwargs.get('relative_d', True)
    self.drive_eye = kwargs.get('drive_eye', None)
    self.delta_eye_arr = kwargs.get('delta_eye_arr', None)
    self.delta_eye_open_n = kwargs.get('delta_eye_open_n', 0)
    self.fade_type = kwargs.get('fade_type', '')
    self.fade_out_keys = kwargs.get('fade_out_keys', ('exp',))
    self.flag_stitching = kwargs.get('flag_stitching', True)

    self.ctrl_info = kwargs.get('ctrl_info', dict())
    self.overall_ctrl_info = kwargs.get('overall_ctrl_info', dict())

    assert self.wav2feat.support_streaming or not self.online_mode

    # ======== Use cached source_info (SKIP avatar_registrar) ========
    self.source_info = source_info
    self.source_info_frames = len(source_info['x_s_info_lst'])
    self.num_source_frames = self.source_info_frames  # refactored pipeline alias

    # ======== Setup Condition Handler ========
    self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

    # ======== Setup Audio2Motion (LMDM) ========
    x_s_info_0 = self.condition_handler.x_s_info_0
    self.audio2motion.setup(
        x_s_info_0,
        overlap_v2=self.overlap_v2,
        fix_kp_cond=self.fix_kp_cond,
        fix_kp_cond_dim=self.fix_kp_cond_dim,
        sampling_timesteps=self.sampling_timesteps,
        online_mode=self.online_mode,
        v_min_max_for_clip=self.v_min_max_for_clip,
        smo_k_d=self.smo_k_d,
    )

    # ======== Setup Motion Stitch ========
    is_image_flag = source_info['is_image_flag']
    x_s_info = source_info['x_s_info_lst'][0]
    self.motion_stitch.setup(
        N_d=self.N_d,
        use_d_keys=self.use_d_keys,
        relative_d=self.relative_d,
        drive_eye=self.drive_eye,
        delta_eye_arr=self.delta_eye_arr,
        delta_eye_open_n=self.delta_eye_open_n,
        fade_out_keys=self.fade_out_keys,
        fade_type=self.fade_type,
        flag_stitching=self.flag_stitching,
        is_image_flag=is_image_flag,
        x_s_info=x_s_info,
        d0=None,
        ch_info=self.ch_info,
        overall_ctrl_info=self.overall_ctrl_info,
    )

    # ======== Video Writer ========
    output_queue = kwargs.get('_output_queue', None)
    if output_queue is not None:
        # Streaming mode: push frames to queue instead of file
        self.output_path = None
        self.tmp_output_path = None
        self.writer = _QueueWriter(output_queue)
        self.writer_pbar = tqdm(desc='streaming-offline')
        self.writer_progress = self.writer_pbar  # refactored pipeline alias
    else:
        self.output_path = output_path
        self.tmp_output_path = output_path + '.tmp.mp4'
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_pbar = tqdm(desc='writer')
        self.writer_progress = self.writer_pbar  # refactored pipeline alias

    # ======== Audio Feat Buffer ========
    if self.online_mode:
        self.audio_feat = self.wav2feat.wav2feat(np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000)
        assert len(self.audio_feat) == self.overlap_v2, f'{len(self.audio_feat)}'
    else:
        self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
    self.cond_idx_start = 0 - len(self.audio_feat)

    # ======== Setup Worker Threads ========
    QUEUE_MAX_SIZE = 100
    self.worker_exception = None
    self.stop_event = threading.Event()

    self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

    self.thread_list = [
        threading.Thread(target=self.audio2motion_worker),
        threading.Thread(target=self.gpu_worker),   # merged: stitch + warp + decode
        threading.Thread(target=self.putback_worker),
        threading.Thread(target=self.writer_worker),
    ]

    for thread in self.thread_list:
        thread.start()


class _QueueWriter:
    """Drop-in replacement for VideoWriterByImageIO that pushes RGB frames to a queue.

    The pipeline's _writer_worker calls self.writer(frame, fmt='rgb') for each frame
    and self.writer.close() when done.  This adapter forwards frames to the output
    queue and sends a None sentinel on close so consumers know the stream ended.
    """
    def __init__(self, output_queue):
        self._queue = output_queue

    def __call__(self, frame, fmt='rgb'):
        self._queue.put(frame)

    def close(self):
        self._queue.put(None)


_ditto_load_lock = threading.Lock()


def load_ditto():
    """Load Ditto SDK (thread-safe). Set DITTO_BACKEND=trt to use TensorRT engines."""
    global ditto_sdk
    if ditto_sdk is None:
        with _ditto_load_lock:
            if ditto_sdk is None:  # double-check after acquiring lock
                from stream_pipeline_offline import StreamSDK
                StreamSDK.setup_from_cache = _setup_from_cache

                backend = os.environ.get("DITTO_BACKEND", "onnx")
                cfg_name, data_dir = _resolve_backend(backend)

                cfg_pkl = str(DITTO_PATH / "checkpoints" / "ditto_cfg" / cfg_name)
                data_root = str(DITTO_PATH / "checkpoints" / data_dir)
                logger.info(f"Loading Ditto SDK (backend={backend}, cfg={cfg_name})...")
                ditto_sdk = StreamSDK(cfg_pkl=cfg_pkl, data_root=data_root)
                logger.info("Ditto SDK loaded!")
    return ditto_sdk


def warmup_offline_pipeline():
    """Run a dummy generation through the offline pipeline to trigger torch.compile.

    Without this, the first /generate request takes ~11s extra for compilation.
    """
    import time

    sdk = load_ditto()
    if not avatar_cache:
        logger.info("[warmup-offline] No cached avatars — skipping offline warmup")
        return

    avatar_id = next(iter(avatar_cache))
    source_info = avatar_cache[avatar_id]

    logger.info(f"[warmup-offline] Running dummy offline pipeline for torch.compile warmup (avatar={avatar_id})...")
    t0 = time.time()

    tmp_output = "/tmp/ditto_warmup_offline.mp4"

    sdk.setup_from_cache(
        source_info, tmp_output,
        sampling_timesteps=DEFAULT_SAMPLING_TIMESTEPS,
        fps=DEFAULT_FPS,
    )

    # Generate 1 second of silence — just enough to trigger all pipeline stages
    silence = np.zeros((16000,), dtype=np.float32)
    num_f = math.ceil(len(silence) / 16000 * DEFAULT_FPS)
    sdk.setup_Nd(N_d=num_f)

    aud_feat = sdk.wav2feat.wav2feat(silence)
    sdk.audio2motion_queue.put(aud_feat)
    sdk.close()

    elapsed = time.time() - t0
    logger.info(f"[warmup-offline] Done in {elapsed:.1f}s — offline torch.compile warmed up")

    # Clean up temp file
    for p in [tmp_output, tmp_output + ".tmp.mp4"]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def _register_avatar(sdk, image_path: str, avatar_id: str | None = None) -> tuple[str, dict]:
    """Run avatar registration and cache the result."""
    if avatar_id is None:
        avatar_id = _image_hash(image_path)

    # Check memory cache first
    with avatar_cache_lock:
        if avatar_id in avatar_cache:
            logger.info(f"Avatar {avatar_id} already cached (memory)")
            return avatar_id, avatar_cache[avatar_id]

    # Check disk cache
    cached = _load_cache(avatar_id)
    if cached is not None:
        with avatar_cache_lock:
            avatar_cache[avatar_id] = cached
        logger.info(f"Avatar {avatar_id} loaded from disk cache")
        return avatar_id, cached

    # Full registration (expensive - runs 4 models)
    logger.info(f"Registering new avatar {avatar_id} from {image_path}...")
    t0 = time.time()
    source_info = sdk.avatar_registrar.register(image_path, max_dim=512)
    elapsed = time.time() - t0
    logger.info(f"Registration complete in {elapsed:.2f}s")

    # Cache in memory + disk
    with avatar_cache_lock:
        avatar_cache[avatar_id] = source_info
    _save_cache(avatar_id, source_info)

    # Persist source image for background pre-render
    persistent_img = _avatar_image_path(avatar_id)
    if not persistent_img.exists():
        shutil.copy2(image_path, persistent_img)

    logger.info(f"Avatar {avatar_id} cached (memory + disk, ~{_estimate_size_mb(source_info):.1f} MB)")

    return avatar_id, source_info


def _avatar_image_path(avatar_id: str) -> Path:
    """Persistent path for an avatar's source image."""
    return IMAGES_DIR / f"{avatar_id}.png"


def _clip_path(avatar_id: str, clip_type: str) -> Path:
    return CLIPS_DIR / f"{avatar_id}_{clip_type}.mp4"


def _clips_ready(avatar_id: str) -> dict[str, bool]:
    """Check which pre-rendered clips exist for an avatar (variant or legacy naming)."""
    result = {}
    for ct in CLIP_TYPES:
        has_variants = len(list(CLIPS_DIR.glob(f"{avatar_id}_{ct}_*.mp4"))) > 0
        has_legacy = _clip_path(avatar_id, ct).exists()
        result[ct] = has_variants or has_legacy
    return result


def _ensure_skyreels_setup(avatar_id: str) -> bool:
    """Run SkyReels-A1 setup.sh if .setup_done marker is missing. Returns True on success."""
    setup_marker = SKYREELS_PATH / ".setup_done"
    if setup_marker.exists():
        return True

    setup_script = SKYREELS_PATH / "setup.sh"
    if not setup_script.exists():
        logger.info(f"[prerender] ERROR: {setup_script} not found — cannot auto-setup SkyReels-A1")
        with prerender_jobs_lock:
            prerender_jobs[avatar_id] = {"status": "failed", "error": "setup.sh not found"}
        return False

    with prerender_jobs_lock:
        prerender_jobs[avatar_id] = {"status": "setting_up", "started": time.time()}
    logger.info(f"[prerender] Running one-time SkyReels-A1 setup (may take 15-20 min)...")

    try:
        result = subprocess.run(
            ["bash", str(setup_script)],
            cwd=str(SKYREELS_PATH),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max for setup
        )
        if result.returncode != 0:
            logger.info(f"[prerender] Setup failed:\n{result.stderr}")
            with prerender_jobs_lock:
                prerender_jobs[avatar_id] = {"status": "failed", "error": f"setup failed: {result.stderr[-1000:]}"}
            return False
        logger.info(f"[prerender] SkyReels-A1 setup complete.")
        return True
    except subprocess.TimeoutExpired:
        with prerender_jobs_lock:
            prerender_jobs[avatar_id] = {"status": "failed", "error": "setup timeout (1hr)"}
        return False
    except Exception as e:
        with prerender_jobs_lock:
            prerender_jobs[avatar_id] = {"status": "failed", "error": f"setup error: {e}"}
        return False


def _run_prerender(avatar_id: str, image_path: str):
    """Run SkyReels-A1 pre-render in a background thread. Updates prerender_jobs state."""
    try:
        # Ensure SkyReels-A1 environment is set up (one-time)
        if not _ensure_skyreels_setup(avatar_id):
            return

        with prerender_jobs_lock:
            prerender_jobs[avatar_id] = {"status": "running", "started": time.time()}
        logger.info(f"[prerender] Starting background clip generation for {avatar_id}")

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
        result = subprocess.run(
            cmd,
            cwd=str(SKYREELS_PATH),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
        )

        if result.returncode == 0:
            with prerender_jobs_lock:
                prerender_jobs[avatar_id] = {"status": "done", "finished": time.time()}
            logger.info(f"[prerender] Clips ready for {avatar_id}")
        else:
            with prerender_jobs_lock:
                prerender_jobs[avatar_id] = {
                    "status": "failed",
                    "error": result.stderr[-1000:] if result.stderr else "unknown error",
                }
            logger.info(f"[prerender] Failed for {avatar_id}:\n{result.stderr}")

    except subprocess.TimeoutExpired:
        with prerender_jobs_lock:
            prerender_jobs[avatar_id] = {"status": "failed", "error": "timeout (30min)"}
        logger.info(f"[prerender] Timeout for {avatar_id}")
    except Exception as e:
        with prerender_jobs_lock:
            prerender_jobs[avatar_id] = {"status": "failed", "error": str(e)}
        logger.info(f"[prerender] Error for {avatar_id}: {e}")


def _start_prerender(avatar_id: str, image_path: str):
    """Submit a pre-render job to the background executor."""
    with prerender_jobs_lock:
        if avatar_id in prerender_jobs and prerender_jobs[avatar_id]["status"] in ("running", "done"):
            return  # already running or finished

    # Save image persistently so the background process can access it
    persistent_path = _avatar_image_path(avatar_id)
    if not persistent_path.exists():
        shutil.copy2(image_path, persistent_path)

    with prerender_jobs_lock:
        prerender_jobs[avatar_id] = {"status": "pending"}
    prerender_executor.submit(_run_prerender, avatar_id, str(persistent_path))


def _estimate_size_mb(source_info: dict) -> float:
    """Rough estimate of source_info memory footprint."""
    total = 0
    for v in source_info.get("f_s_lst", []):
        total += v.nbytes if hasattr(v, 'nbytes') else 0
    for v in source_info.get("img_rgb_lst", []):
        total += v.nbytes if hasattr(v, 'nbytes') else 0
    return total / (1024 * 1024)


# ---------- Pre-rendered clip playback ----------

# Decoded clip frames cached in memory with LRU eviction to prevent OOM.
# Max 20 clips (~1GB at 512x340 resolution). Oldest-accessed evicted first.
_CLIP_CACHE_MAX = int(os.environ.get("CLIP_CACHE_MAX", "20"))
_clip_frame_cache: dict[str, list[np.ndarray]] = {}
_clip_cache_order: list[str] = []  # LRU order: most-recently-used at end


def _clip_cache_put(key: str, frames: list[np.ndarray]):
    """Insert into clip cache with LRU eviction."""
    if key in _clip_frame_cache:
        _clip_cache_order.remove(key)
    _clip_frame_cache[key] = frames
    _clip_cache_order.append(key)
    # Evict oldest entries if over limit
    while len(_clip_cache_order) > _CLIP_CACHE_MAX:
        evict_key = _clip_cache_order.pop(0)
        _clip_frame_cache.pop(evict_key, None)


def _clip_cache_get(key: str) -> list[np.ndarray] | None:
    """Get from clip cache and mark as recently used."""
    if key in _clip_frame_cache:
        _clip_cache_order.remove(key)
        _clip_cache_order.append(key)
        return _clip_frame_cache[key]
    return None


def _get_clip_variants(avatar_id: str, clip_type: str) -> list[Path]:
    """Find all variant clip files for an avatar + clip type."""
    variants = sorted(CLIPS_DIR.glob(f"{avatar_id}_{clip_type}_*.mp4"))
    # Also support legacy single-file naming (no variant index)
    legacy = CLIPS_DIR / f"{avatar_id}_{clip_type}.mp4"
    if legacy.exists() and legacy not in variants:
        variants.insert(0, legacy)
    return variants


def _decode_clip_frames(clip_path: Path) -> list[np.ndarray]:
    """Decode an mp4 into a list of RGB uint8 numpy arrays."""
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _crop_resize_to_fill(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Crop frame to target aspect ratio (center-crop) then resize. No black bars."""
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # Source is wider — crop sides
        new_w = int(h * target_ratio)
        x_off = (w - new_w) // 2
        frame = frame[:, x_off:x_off + new_w]
    elif src_ratio < target_ratio:
        # Source is taller — crop top/bottom
        new_h = int(w / target_ratio)
        y_off = (h - new_h) // 2
        frame = frame[y_off:y_off + new_h, :]

    return cv2.resize(frame, (target_w, target_h))


def _load_avatar_clips(avatar_id: str, clip_type: str = "idle") -> list[list[np.ndarray]]:
    """Load all variant clips for an avatar into memory. Returns list of frame sequences.

    Clips are resized to match the avatar's output resolution so there's
    no jarring switch when transitioning between idle clips and Ditto output.
    """
    variants = _get_clip_variants(avatar_id, clip_type)
    result = []
    for clip_path in variants:
        cache_key = str(clip_path)
        cached = _clip_cache_get(cache_key)
        if cached is None:
            frames = _decode_clip_frames(clip_path)
            if frames:
                # Crop and resize to match avatar output resolution
                with avatar_cache_lock:
                    ref_img = avatar_cache[avatar_id]["img_rgb_lst"][0]
                target_h, target_w = ref_img.shape[:2]
                if frames[0].shape[:2] != (target_h, target_w):
                    frames = [_crop_resize_to_fill(f, target_w, target_h) for f in frames]
                _clip_cache_put(cache_key, frames)
                cached = frames
                logger.info(f"[clips] Loaded {clip_path.name}: {len(frames)} frames")
        if cached is not None:
            result.append(cached)
    return result


def _generate_video(sdk, avatar_id: str, audio_path: str,
                    sampling_timesteps: int = None, fps: int = None) -> bytes:
    """Generate video using cached avatar."""
    if avatar_id not in avatar_cache:
        raise ValueError(f"Avatar {avatar_id} not found in cache")

    timesteps = sampling_timesteps or DEFAULT_SAMPLING_TIMESTEPS
    video_fps = fps or DEFAULT_FPS
    source_info = avatar_cache[avatar_id]

    tmp_output = "/tmp/ditto_tmp.mp4"
    # Pipeline writes raw video to output_path + ".tmp.mp4"
    raw_video = tmp_output + ".tmp.mp4"

    # Use cached source_info — skips registration entirely
    t0 = time.time()
    sdk.setup_from_cache(
        source_info, tmp_output,
        sampling_timesteps=timesteps,
        fps=video_fps,
    )
    setup_time = time.time() - t0

    # Load and process audio
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * video_fps)

    sdk.setup_Nd(N_d=num_f)

    t1 = time.time()
    aud_feat = sdk.wav2feat.wav2feat(audio)
    sdk.audio2motion_queue.put(aud_feat)
    sdk.close()
    gen_time = time.time() - t1

    logger.info(f"Setup: {setup_time:.3f}s | Generation: {gen_time:.3f}s | Total: {setup_time + gen_time:.3f}s "
          f"(timesteps={timesteps}, fps={video_fps})")

    # Return raw video directly — skip ffmpeg audio mux since
    # the LiveKit client handles audio separately
    with open(raw_video, "rb") as f:
        return f.read()


def _generate_video_from_text(sdk, avatar_id: str, text: str, voice: str = "tara") -> bytes:
    """Generate video from text - streams TTS then processes through Ditto."""
    if avatar_id not in avatar_cache:
        raise ValueError(f"Avatar {avatar_id} not found in cache")

    source_info = avatar_cache[avatar_id]

    tmp_output = "/tmp/ditto_pipeline_tmp.mp4"
    final_output = "/tmp/ditto_pipeline_output.mp4"
    audio_output = "/tmp/ditto_pipeline_audio.wav"

    t0 = time.time()

    # 1. Stream TTS audio and accumulate
    all_audio_24k = bytearray()
    header_skipped = False
    header_buf = bytearray()

    client = _get_tts_client()
    with client.stream("POST", TTS_URL, json={
        "text": text,
        "voice": voice,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 8000,
        "repetition_penalty": 1.1,
    }) as response:
        response.raise_for_status()
        for raw_chunk in response.iter_bytes(chunk_size=4096):
            if not header_skipped:
                header_buf.extend(raw_chunk)
                if len(header_buf) >= WAV_HEADER_SIZE:
                    all_audio_24k.extend(header_buf[WAV_HEADER_SIZE:])
                    header_skipped = True
                continue
            all_audio_24k.extend(raw_chunk)

    tts_time = time.time() - t0

    # 2. Convert and resample: 24kHz int16 -> 16kHz float32
    audio_24k = np.frombuffer(bytes(all_audio_24k), dtype=np.int16).astype(np.float32) / 32768.0
    audio_16k = resample_poly(audio_24k, up=2, down=3).astype(np.float32)
    audio_duration = len(audio_24k) / TTS_SAMPLE_RATE
    num_f = math.ceil(len(audio_16k) / DITTO_SAMPLE_RATE * DEFAULT_FPS)
    logger.info(f"[pipeline] TTS: {tts_time:.3f}s for {audio_duration:.1f}s audio")

    # 3. Save audio for ffmpeg merge
    sf.write(audio_output, audio_24k, TTS_SAMPLE_RATE)

    # 4. Setup Ditto and process (offline mode - proven stable)
    t1 = time.time()
    sdk.setup_from_cache(
        source_info, tmp_output,
        sampling_timesteps=DEFAULT_SAMPLING_TIMESTEPS,
        fps=DEFAULT_FPS,
    )
    sdk.setup_Nd(N_d=num_f)

    aud_feat = sdk.wav2feat.wav2feat(audio_16k)
    sdk.audio2motion_queue.put(aud_feat)
    sdk.close()
    ditto_time = time.time() - t1
    logger.info(f"[pipeline] Ditto: {ditto_time:.3f}s")

    # 5. Merge video + audio (needed for /generate_from_text since client plays MP4 directly)
    raw_video = tmp_output + ".tmp.mp4"
    ffmpeg_result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", raw_video, "-i", audio_output,
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", final_output],
        capture_output=True, text=True,
    )
    if ffmpeg_result.returncode != 0:
        logger.info(f"[pipeline] ffmpeg merge failed: {ffmpeg_result.stderr[:500]}")

    for f_path in [raw_video, audio_output]:
        if os.path.exists(f_path):
            os.remove(f_path)

    total_time = time.time() - t0
    rtf = total_time / audio_duration if audio_duration > 0 else 0
    logger.info(f"[pipeline] Total: {total_time:.3f}s (TTS {tts_time:.1f}s + Ditto {ditto_time:.1f}s) RTF: {rtf:.2f}x")

    with open(final_output, "rb") as f:
        return f.read()

def _validate_config():
    """Validate critical configuration at startup. Fail fast on misconfiguration."""
    issues = []
    if not DITTO_PATH.exists():
        issues.append(f"DITTO_PATH does not exist: {DITTO_PATH}")
    if not CACHE_DIR.exists():
        issues.append(f"AVATAR_CACHE_DIR does not exist: {CACHE_DIR}")
    if TTS_URL and not TTS_URL.startswith(("http://", "https://")):
        issues.append(f"TTS_URL is not a valid URL: {TTS_URL}")
    if issues:
        for issue in issues:
            logger.info(f"[config] WARNING: {issue}")
    else:
        logger.info("[config] All configuration paths validated.")


@app.on_event("startup")
async def startup():
    """Load model, warm-load cached avatars, detect existing clips."""
    _validate_config()
    logger.info("Warming up avatar cache from disk...")
    _warm_load_all()

    # Detect pre-rendered clips for cached avatars and pre-load all types
    for avatar_id in avatar_cache:
        clips_status = _clips_ready(avatar_id)
        if any(clips_status.values()):
            prerender_jobs[avatar_id] = {"status": "done"}
            total_variants = 0
            total_frames = 0
            for ct in CLIP_TYPES:
                if clips_status.get(ct):
                    loaded = _load_avatar_clips(avatar_id, ct)
                    total_variants += len(loaded)
                    total_frames += sum(len(c) for c in loaded)
            logger.info(f"  Clips loaded for {avatar_id}: {total_variants} variants, {total_frames} frames")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_ditto)

    # Warm up offline pipeline (used for both /generate and streaming now)
    await loop.run_in_executor(None, warmup_offline_pipeline)

    # Start session cleanup background task
    asyncio.create_task(_session_cleanup_loop())


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown: stop all sessions, cancel background jobs, cleanup."""
    logger.info("[shutdown] Stopping all active sessions...")
    for sid in list(active_sessions.keys()):
        try:
            await _auto_stop_session(sid)
        except Exception as e:
            logger.info(f"[shutdown] Error stopping session {sid}: {e}")
    logger.info("[shutdown] Shutting down prerender executor...")
    prerender_executor.shutdown(wait=False)
    logger.info("[shutdown] Cleanup complete.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "ditto",
        "cached_avatars": len(avatar_cache),
        "avatar_ids": list(avatar_cache.keys()),
        "sampling_timesteps": DEFAULT_SAMPLING_TIMESTEPS,
        "fps": DEFAULT_FPS,
    }


@app.get("/avatars")
async def list_avatars(limit: int = 50, offset: int = 0):
    """List cached avatars with pagination. ?limit=50&offset=0"""
    with avatar_cache_lock:
        all_ids = list(avatar_cache.keys())
    total = len(all_ids)
    page_ids = all_ids[offset:offset + limit]
    avatars = []
    for avatar_id in page_ids:
        with avatar_cache_lock:
            info = avatar_cache.get(avatar_id)
        if info is None:
            continue
        avatars.append({
            "avatar_id": avatar_id,
            "frames": len(info.get("x_s_info_lst", [])),
            "is_image": info.get("is_image_flag", True),
            "size_mb": round(_estimate_size_mb(info), 1),
            "on_disk": _cache_path(avatar_id).exists(),
            "inference_ready": True,
            "clips_ready": _clips_ready(avatar_id),
            "prerender_status": prerender_jobs.get(avatar_id, {}).get("status", "none"),
        })
    return {"avatars": avatars, "total": total, "limit": limit, "offset": offset}


@app.delete("/avatars/{avatar_id}")
async def delete_avatar(avatar_id: str):
    """Evict an avatar from memory, disk cache, and clips."""
    with avatar_cache_lock:
        removed_memory = avatar_id in avatar_cache
        if removed_memory:
            del avatar_cache[avatar_id]

    removed_disk = False
    disk_path = _cache_path(avatar_id)
    if disk_path.exists():
        disk_path.unlink()
        removed_disk = True

    # Clean up clips (variant + legacy naming) and stored image
    for ct in CLIP_TYPES:
        for variant_file in CLIPS_DIR.glob(f"{avatar_id}_{ct}_*.mp4"):
            variant_file.unlink()
        legacy = _clip_path(avatar_id, ct)
        if legacy.exists():
            legacy.unlink()
    # Evict decoded frames from memory
    for key in list(_clip_frame_cache.keys()):
        if avatar_id in key:
            _clip_frame_cache.pop(key, None)
            if key in _clip_cache_order:
                _clip_cache_order.remove(key)
    img_path = _avatar_image_path(avatar_id)
    if img_path.exists():
        img_path.unlink()

    with prerender_jobs_lock:
        prerender_jobs.pop(avatar_id, None)

    if not removed_memory and not removed_disk:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    return {"status": "deleted", "avatar_id": avatar_id}


@app.get("/avatars/{avatar_id}/status")
async def avatar_status(avatar_id: str):
    """Detailed status for a single avatar."""
    if avatar_id not in avatar_cache:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    return {
        "avatar_id": avatar_id,
        "inference_ready": True,
        "clips_ready": _clips_ready(avatar_id),
        "prerender": prerender_jobs.get(avatar_id, {"status": "none"}),
    }


@app.get("/clips/{avatar_id}/{clip_type}")
async def get_clip(avatar_id: str, clip_type: str):
    """Serve a pre-rendered clip (idle, thinking, greeting). Returns a random variant."""
    if clip_type not in CLIP_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid clip type. Choose from: {CLIP_TYPES}")

    variants = _get_clip_variants(avatar_id, clip_type)
    if not variants:
        job = prerender_jobs.get(avatar_id, {})
        status = job.get("status", "none")
        if status == "running":
            raise HTTPException(status_code=202, detail=f"Clip '{clip_type}' is still rendering")
        elif status == "failed":
            raise HTTPException(status_code=500, detail=f"Clip render failed: {job.get('error', 'unknown')}")
        else:
            raise HTTPException(status_code=404, detail=f"Clip '{clip_type}' not found for avatar {avatar_id}")

    path = random.choice(variants)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.post("/avatars/{avatar_id}/prerender")
async def trigger_prerender(avatar_id: str):
    """Manually trigger clip pre-rendering for an avatar."""
    if avatar_id not in avatar_cache:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    img_path = _avatar_image_path(avatar_id)
    if not img_path.exists():
        raise HTTPException(status_code=400, detail="Source image not stored. Re-register the avatar.")

    _start_prerender(avatar_id, str(img_path))

    return {
        "avatar_id": avatar_id,
        "prerender_status": "submitted",
    }


@app.post("/register")
async def register_avatar(request: RegisterAvatarRequest):
    """Register an avatar image. Caches features for instant reuse."""
    sdk = load_ditto()

    try:
        if request.image_base64:
            if len(request.image_base64) > MAX_BASE64_SIZE:
                raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_BASE64_SIZE // 1024 // 1024}MB)")
            img_data = base64.b64decode(request.image_base64)
            temp_path = "/tmp/avatar_temp.png"
            with open(temp_path, "wb") as f:
                f.write(img_data)
            image_path = temp_path
        elif request.image_path:
            image_path = str(_validate_file_path(request.image_path))
            if not os.path.exists(image_path):
                raise HTTPException(status_code=400, detail="Image not found at provided path")
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        avatar_id, source_info = _register_avatar(sdk, image_path, request.avatar_id)

        # Kick off background clip pre-rendering (non-blocking)
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
        logger.error(f"[error] Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/avatars/{avatar_id}/resolution")
async def avatar_resolution(avatar_id: str):
    """Return the output resolution for an avatar (matches registration crop)."""
    if avatar_id not in avatar_cache:
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")
    ref = avatar_cache[avatar_id]["img_rgb_lst"][0]
    return {"width": ref.shape[1], "height": ref.shape[0]}


@app.post("/generate_frames")
async def generate_frames(request: Request):
    """Accept raw PCM audio, stream back length-prefixed JPEG frames.

    Headers:
      X-Avatar-Id: avatar ID (default: first cached)
      X-Sample-Rate: input sample rate (default: 24000)
      Content-Type: application/octet-stream

    Body: raw int16 PCM audio bytes at X-Sample-Rate

    Response: streaming binary — each frame is [4-byte big-endian length][JPEG bytes]
    """
    import queue as queue_mod

    avatar_id = request.headers.get("X-Avatar-Id", next(iter(avatar_cache), "imogen"))
    input_sr = int(request.headers.get("X-Sample-Rate", "24000"))
    audio_bytes = await request.body()

    if avatar_id not in avatar_cache:
        raise HTTPException(status_code=400, detail=f"Avatar '{avatar_id}' not found")

    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio too short")

    # Resample to 16kHz if needed
    audio_raw = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if input_sr != 16000:
        # resample_poly needs integer up/down factors
        from math import gcd
        g = gcd(16000, input_sr)
        audio_16k = resample_poly(audio_raw.astype(np.float64), 16000 // g, input_sr // g).astype(np.float32)
    else:
        audio_16k = audio_raw

    audio_duration = len(audio_raw) / input_sr
    num_f = math.ceil(len(audio_16k) / 16000 * DEFAULT_FPS)
    logger.info(f"[generate_frames] avatar={avatar_id}, input_sr={input_sr}, "
          f"duration={audio_duration:.2f}s, expected_frames={num_f}")

    sdk = load_ditto()
    source_info = avatar_cache[avatar_id]

    # Setup offline pipeline with queue-based writer
    output_queue = queue_mod.Queue(maxsize=200)
    sdk.setup_from_cache(
        source_info, "/dev/null",
        _output_queue=output_queue,
        sampling_timesteps=DEFAULT_SAMPLING_TIMESTEPS,
        fps=DEFAULT_FPS,
    )
    sdk.setup_Nd(N_d=num_f)

    aud_feat = sdk.wav2feat.wav2feat(audio_16k, sr=16000)
    sdk.audio2motion_queue.put(aud_feat)

    def frame_generator():
        """Run pipeline close (triggers flush) then yield JPEG frames."""
        # close() blocks until all pipeline workers finish
        sdk.close()

        frame_count = 0
        while True:
            try:
                frame = output_queue.get(timeout=30)
            except queue_mod.Empty:
                break
            if frame is None:
                break
            # Encode RGB frame as JPEG
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpeg_bytes = jpeg.tobytes()
            yield struct.pack(">I", len(jpeg_bytes)) + jpeg_bytes
            frame_count += 1

        logger.info(f"[generate_frames] Streamed {frame_count} frames for avatar={avatar_id}")

    return StreamingResponse(frame_generator(), media_type="application/octet-stream")


@app.post("/generate_from_text")
async def generate_from_text(request: TextGenerateRequest):
    """Generate video from text - pipelines TTS + Ditto for lower latency."""
    sdk = load_ditto()

    if request.avatar_id not in avatar_cache:
        raise HTTPException(status_code=400, detail=f"Avatar '{request.avatar_id}' not found. Call /register first.")

    try:
        video_data = _generate_video_from_text(sdk, request.avatar_id, request.text, request.voice)
        return Response(content=video_data, media_type="video/mp4")
    except Exception as e:
        logger.error(f"[error] Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/generate")
async def generate_video(request: GenerateRequest):
    """Generate video from audio using a cached avatar."""
    sdk = load_ditto()

    if request.avatar_id not in avatar_cache:
        raise HTTPException(status_code=400, detail=f"Avatar '{request.avatar_id}' not found. Call /register first.")

    try:
        if request.audio_base64:
            if len(request.audio_base64) > MAX_BASE64_SIZE:
                raise HTTPException(status_code=413, detail=f"Audio too large (max {MAX_BASE64_SIZE // 1024 // 1024}MB)")
            audio_data = base64.b64decode(request.audio_base64)
            audio_path = "/tmp/audio_temp.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_data)
        elif request.audio_path:
            audio_path = str(_validate_file_path(request.audio_path))
        else:
            raise HTTPException(status_code=400, detail="No audio provided")

        video_data = _generate_video(
            sdk, request.avatar_id, audio_path,
            sampling_timesteps=request.sampling_timesteps,
            fps=request.fps,
        )

        return Response(
            content=video_data,
            media_type="video/mp4"
        )

    except Exception as e:
        logger.error(f"[error] Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _rgb_to_rgba(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to RGBA by adding alpha=255 channel."""
    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


# Clip types to mix during idle periods — weighted so idle is most common
_IDLE_MIX_TYPES = ["idle", "idle", "idle", "thinking", "lookingup", "lippurse"]


async def _unified_publisher(session_id: str, session: dict):
    """Publishes ALL frames (idle + speech) using AVSynchronizer + perf_counter pacing.

    Uses rtc.AVSynchronizer for A/V sync and RTP timestamps.
    Uses time.perf_counter() for precise frame pacing (sub-ms accuracy).
    Audio pushed through av_sync first, then video — AVSync handles the rest.
    """
    import queue as queue_mod
    import time as _time

    video_source = session["video_source"]
    audio_source = session["audio_source"]
    av_sync = session["av_sync"]
    fps = session["fps"]
    pub_w = session["publish_w"]
    pub_h = session["publish_h"]
    avatar_id = session["avatar_id"]
    frame_interval = 1.0 / fps  # 0.04s at 25fps

    AUDIO_SAMPLE_RATE = 24000
    AUDIO_SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // fps  # 960 at 25fps

    # RGBA conversion buffer (reused)
    _rgba_buf = None

    def _make_vf(rgb):
        nonlocal _rgba_buf
        h, w = rgb.shape[:2]
        if h != pub_h or w != pub_w:
            rgb = cv2.resize(rgb, (pub_w, pub_h), interpolation=cv2.INTER_AREA)
        h, w = pub_h, pub_w
        if _rgba_buf is None or _rgba_buf.shape[0] != h or _rgba_buf.shape[1] != w:
            _rgba_buf = np.empty((h, w, 4), dtype=np.uint8)
            _rgba_buf[:, :, 3] = 255
        _rgba_buf[:, :, :3] = rgb
        return lk_rtc.VideoFrame(
            width=w, height=h,
            type=lk_rtc.VideoBufferType.RGBA,
            data=_rgba_buf.tobytes(),
        )

    def _get_audio_buf():
        """Consolidate dirty audio chunks and return the buffer."""
        if session.get("audio_24k_dirty"):
            session["audio_24k_buffer"] = np.concatenate(session["audio_24k_chunks"])
            session["audio_24k_dirty"] = False
        return session.get("audio_24k_buffer")

    async def _precise_sleep(target_time):
        """High-precision sleep using perf_counter — yields to event loop."""
        remaining = target_time - _time.perf_counter()
        if remaining > 0.002:
            await asyncio.sleep(remaining - 0.001)  # sleep most of it
        # Busy-wait for final ~1ms precision
        while _time.perf_counter() < target_time:
            await asyncio.sleep(0)

    idle_frame_count = 0
    speech_frame_count = 0
    loop = asyncio.get_event_loop()

    # Static reference image (fallback when no clips available)
    ref_img = avatar_cache[avatar_id]["img_rgb_lst"][0]

    # Pre-load idle clips — build a flat playlist cycling through mix types
    # Each entry is a list of frames (one clip variant)
    _idle_playlist: list[list[np.ndarray]] = []
    for clip_type in _IDLE_MIX_TYPES:
        variants = _load_avatar_clips(avatar_id, clip_type)
        _idle_playlist.extend(variants)

    _idle_clip_idx = 0       # which clip in the playlist
    _idle_frame_idx = 0      # which frame within that clip
    _has_idle_clips = len(_idle_playlist) > 0
    if _has_idle_clips:
        import random as _random
        _random.shuffle(_idle_playlist)
        logger.info(f"[publisher] Loaded {len(_idle_playlist)} idle clip variants for {avatar_id}")

    logger.info(f"[publisher] Started unified publisher for session={session_id} fps={fps} (AVSync + perf_counter)")

    try:
        while session_id in active_sessions:
            output_queue = session.get("output_queue")

            if output_queue is None:
                # ── IDLE MODE ── (paced to fps)
                if idle_frame_count == 0:
                    _idle_t0 = _time.perf_counter()

                target_t = _idle_t0 + (idle_frame_count + 1) * frame_interval
                await _precise_sleep(target_t)

                # Push silent audio + video via AVSync — AVSync stalls when
                # video-only is pushed (waits ~2s for audio). Pushing silence
                # keeps AVSync fed and prevents "behind schedule" warnings.
                _silent_af = lk_rtc.AudioFrame(
                    data=bytes(AUDIO_SAMPLES_PER_FRAME * 2),  # int16 = 2 bytes each
                    sample_rate=AUDIO_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=AUDIO_SAMPLES_PER_FRAME,
                )
                if _has_idle_clips:
                    clip = _idle_playlist[_idle_clip_idx % len(_idle_playlist)]
                    frame_rgb = clip[_idle_frame_idx % len(clip)]
                    _idle_frame_idx += 1
                    if _idle_frame_idx >= len(clip):
                        _idle_frame_idx = 0
                        _idle_clip_idx += 1  # advance to next clip when current ends
                    await av_sync.push(_silent_af)
                    await av_sync.push(_make_vf(frame_rgb))
                else:
                    await av_sync.push(_silent_af)
                    await av_sync.push(_make_vf(ref_img))
                idle_frame_count += 1
                if idle_frame_count <= 3 or idle_frame_count % 200 == 0:
                    mode = "clip" if _has_idle_clips else "static"
                    logger.info(f"[publisher] idle frame #{idle_frame_count} ({mode})")

            else:
                # ── SPEECH MODE: precise pacing with AVSync ──
                _speech_t0_perf = _time.perf_counter()
                _speech_t0_mono = _time.monotonic()
                logger.info(f"[publisher] Entering speech mode (AVSync + perf_counter, fps={fps})")

                audio_sample_offset = 0
                speech_frame_count = 0
                _push_times = []
                _frame_arrive_times = []  # When frames arrived from Ditto pipeline

                try:
                    while True:
                        if session_id not in active_sessions:
                            break

                        try:
                            frame = await loop.run_in_executor(
                                None, lambda: output_queue.get(timeout=frame_interval)
                            )
                        except queue_mod.Empty:
                            if session.get("_speech_interrupted"):
                                break
                            # Pipeline not ready yet — push silent audio + ref_img
                            # via AVSync to keep it fed during LMDM startup (~1-2s)
                            _fill_af = lk_rtc.AudioFrame(
                                data=bytes(AUDIO_SAMPLES_PER_FRAME * 2),
                                sample_rate=AUDIO_SAMPLE_RATE,
                                num_channels=1,
                                samples_per_channel=AUDIO_SAMPLES_PER_FRAME,
                            )
                            await av_sync.push(_fill_af)
                            await av_sync.push(_make_vf(ref_img))
                            continue
                        if frame is None:
                            break

                        # If interrupted, discard frame immediately (drain queue fast so
                        # sdk.close() can unblock its pipeline threads)
                        if session.get("_speech_interrupted"):
                            continue

                        speech_frame_count += 1
                        session["last_ditto_frame"] = frame
                        _frame_arrive_times.append(_time.perf_counter())

                        # Reset t0 on first speech frame
                        if speech_frame_count == 1:
                            _speech_t0_perf = _time.perf_counter()
                            _speech_t0_mono = _time.monotonic()

                        # Precise wall-clock pacing FIRST using perf_counter
                        target_t = _speech_t0_perf + speech_frame_count * frame_interval
                        await _precise_sleep(target_t)

                        t_before = _time.perf_counter()

                        # Push audio through AVSync (delayed by AUDIO_DELAY_FRAMES)
                        # Lip motion leads audio due to HuBERT lookahead — push
                        # video-only frames first, then start audio
                        if speech_frame_count > AUDIO_DELAY_FRAMES:
                            audio_buf = _get_audio_buf()
                            if audio_buf is not None and audio_sample_offset < len(audio_buf):
                                end = min(audio_sample_offset + AUDIO_SAMPLES_PER_FRAME, len(audio_buf))
                                chunk = audio_buf[audio_sample_offset:end]
                                if len(chunk) > 0:
                                    af = lk_rtc.AudioFrame(
                                        data=chunk.tobytes(),
                                        sample_rate=AUDIO_SAMPLE_RATE,
                                        num_channels=1,
                                        samples_per_channel=len(chunk),
                                    )
                                    await av_sync.push(af)
                                    audio_sample_offset = end

                        # Push video through AVSync
                        await av_sync.push(_make_vf(frame))

                        t_after = _time.perf_counter()
                        _push_times.append(t_after - t_before)

                        if speech_frame_count <= 3 or speech_frame_count % 25 == 0:
                            elapsed = _time.monotonic() - _speech_t0_mono
                            push_ms = _push_times[-1] * 1000
                            logger.info(f"[publisher] frame#{speech_frame_count} t={elapsed:.2f}s "
                                  f"push={push_ms:.1f}ms")

                    # Flush remaining audio with last frame held (skip if interrupted)
                    if not session.get("_speech_interrupted"):
                        total_audio = session.get("audio_24k_total_samples", 0)
                        audio_buf = _get_audio_buf()
                        if audio_buf is not None and audio_sample_offset < total_audio:
                            remaining = audio_buf[audio_sample_offset:total_audio]
                            if len(remaining) > 0:
                                last_frame = session.get("last_ditto_frame")
                                if last_frame is not None:
                                    off = 0
                                    while off < len(remaining):
                                        chunk = remaining[off:off + AUDIO_SAMPLES_PER_FRAME]
                                        speech_frame_count += 1
                                        af = lk_rtc.AudioFrame(
                                            data=chunk.tobytes(),
                                            sample_rate=AUDIO_SAMPLE_RATE,
                                            num_channels=1,
                                            samples_per_channel=len(chunk),
                                        )
                                        target_t = _speech_t0_perf + speech_frame_count * frame_interval
                                        await _precise_sleep(target_t)
                                        await av_sync.push(af)
                                        await av_sync.push(_make_vf(last_frame))
                                        off += AUDIO_SAMPLES_PER_FRAME

                        await audio_source.wait_for_playout()
                    else:
                        logger.info(f"[publisher] Interrupted — skipped audio flush and playout wait")

                except Exception as e:
                    logger.info(f"[publisher] Speech error: {e}")
                    import traceback; traceback.print_exc()

                _speech_dur = _time.monotonic() - _speech_t0_mono
                total_audio = session.get("audio_24k_total_samples", 0)
                audio_dur = total_audio / AUDIO_SAMPLE_RATE

                expected_frames = int(audio_dur * fps)
                video_dur = speech_frame_count / fps if fps > 0 else 0
                ratio = video_dur / audio_dur if audio_dur > 0 else 0

                logger.info(f"\n{'='*60}")
                logger.info(f"  PUBLISHER TIMING REPORT")
                logger.info(f"{'='*60}")
                logger.info(f"  Pipeline frames:    {speech_frame_count}")
                logger.info(f"  Expected @{fps}fps:  {expected_frames} frames")
                logger.info(f"  Audio duration:     {audio_dur:.2f}s")
                logger.info(f"  Video duration:     {video_dur:.2f}s (@{fps}fps)")
                logger.info(f"  V/A ratio:          {ratio:.3f}x {'OK' if 0.95 < ratio < 1.05 else 'MISMATCH!'}")
                logger.info(f"  Wall-clock:         {_speech_dur:.2f}s")
                logger.info(f"  Audio delay:        {AUDIO_DELAY_FRAMES} frames ({AUDIO_DELAY_FRAMES * 1000 / fps:.0f}ms)")
                logger.info(f"  Audio pushed:       {audio_sample_offset} / {total_audio} samples")
                if _push_times:
                    logger.info(f"  AVSync push time:   min={min(_push_times)*1000:.1f}ms "
                          f"avg={sum(_push_times)/len(_push_times)*1000:.1f}ms "
                          f"max={max(_push_times)*1000:.1f}ms")
                if len(_frame_arrive_times) > 1:
                    gaps = [_frame_arrive_times[i+1] - _frame_arrive_times[i]
                            for i in range(len(_frame_arrive_times) - 1)]
                    burst = sum(1 for g in gaps if g < 0.005)
                    logger.info(f"  Frame arrival gaps: min={min(gaps)*1000:.1f}ms "
                          f"avg={sum(gaps)/len(gaps)*1000:.1f}ms "
                          f"max={max(gaps)*1000:.1f}ms  "
                          f"burst(<5ms)={burst}/{len(gaps)}")
                logger.info(f"{'='*60}\n")

                # Return to idle — reset idle timer so pacing restarts cleanly
                session["output_queue"] = None
                session["_speech_done_event"].set()
                idle_frame_count = 0

    except asyncio.CancelledError:
        logger.info(f"[publisher] Cancelled (idle={idle_frame_count}, speech={speech_frame_count})")
    except Exception as e:
        logger.info(f"[publisher] ERROR: {e}")
        import traceback; traceback.print_exc()


def _make_idle_frame(avatar_id: str) -> lk_rtc.VideoFrame:
    """Create a VideoFrame from the avatar's reference image for idle display."""
    ref_img = avatar_cache[avatar_id]["img_rgb_lst"][0]  # RGB uint8
    rgba = _rgb_to_rgba(ref_img)
    return lk_rtc.VideoFrame(
        width=ref_img.shape[1],
        height=ref_img.shape[0],
        type=lk_rtc.VideoBufferType.RGBA,
        data=rgba.tobytes(),
    )


async def _audio_receiver_task(session_id: str, session: dict, agent_identity: str):
    """Receive audio from agent via LiveKit ByteStream and feed Ditto pipeline.

    Replaces the old WebSocket /stream handler. Audio arrives on topic "lk.audio_stream"
    from the agent participant, is normalized to 24kHz mono for publishing, then
    resampled to 16kHz for Ditto inference.
    """
    import json as _json
    import queue as queue_mod

    room: lk_rtc.Room = session["room"]
    sdk = load_ditto()  # Use OFFLINE SDK — proven correct speed
    audio_accum = bytearray()
    MAX_AUDIO_SEGMENT_BYTES = 50 * 1024 * 1024  # 50MB max per segment (~9 min at 24kHz 16-bit)
    pipeline_active = False

    # Shared state for interrupt handling
    interrupted = asyncio.Event()
    current_reader_cleared = False

    # Channel for incoming ByteStreamReaders
    reader_queue: asyncio.Queue[lk_rtc.ByteStreamReader] = asyncio.Queue()

    def _parse_int_attr(attrs: dict, key: str, default: int) -> int:
        val = attrs.get(key, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            try:
                return int(float(val))
            except (TypeError, ValueError):
                return default

    def _on_byte_stream(reader: lk_rtc.ByteStreamReader, participant_id: str):
        if participant_id != agent_identity:
            return
        reader_queue.put_nowait(reader)

    def _on_clear_buffer(data: lk_rtc.RpcInvocationData) -> str:
        nonlocal current_reader_cleared
        if data.caller_identity != agent_identity:
            return "reject"
        current_reader_cleared = True
        interrupted.set()
        # Signal publisher to discard remaining frames ASAP
        session["_speech_interrupted"] = True
        return "ok"

    # Register handlers
    room.register_byte_stream_handler(AUDIO_STREAM_TOPIC, _on_byte_stream)
    room.local_participant.register_rpc_method(RPC_CLEAR_BUFFER, _on_clear_buffer)

    logger.info(f"[datastream] Audio receiver started for session={session_id}, agent={agent_identity}")

    try:
        while session_id in active_sessions:
            # Wait for next audio stream from agent
            try:
                reader = await asyncio.wait_for(reader_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

            # Reset state for new segment
            current_reader_cleared = False
            interrupted.clear()
            session["_speech_interrupted"] = False
            audio_accum.clear()

            # Extract stream format from attributes
            attrs = reader.info.attributes or {}
            sender_sr = _parse_int_attr(attrs, "sample_rate", 24000)
            sender_channels = _parse_int_attr(
                attrs,
                "num_channels",
                _parse_int_attr(attrs, "channels", 1),
            )

            logger.info(
                f"[datastream] New audio stream: sample_rate={sender_sr}, "
                f"channels={sender_channels}"
            )

            # Buffer all audio first, then process with offline pipeline at segment end
            session["audio_24k_chunks"] = []
            session["audio_24k_buffer"] = None
            session["audio_24k_total_samples"] = 0
            session["audio_24k_dirty"] = False

            loop = asyncio.get_event_loop()
            total_bytes = 0

            # Read audio data from ByteStream (buffer raw stream first).
            try:
                async for data in reader:
                    if current_reader_cleared:
                        break

                    total_bytes += len(data)

                    # Buffer raw bytes for batch resampling at segment end
                    if len(audio_accum) + len(data) > MAX_AUDIO_SEGMENT_BYTES:
                        logger.info(f"[datastream] WARNING: Audio segment exceeded {MAX_AUDIO_SEGMENT_BYTES // 1024 // 1024}MB limit, truncating")
                        break
                    audio_accum.extend(data)
            except Exception as e:
                logger.info(f"[datastream] Error reading stream: {e}")

            if current_reader_cleared:
                # Handle interruption — no pipeline to tear down since we haven't started one
                logger.info(f"[datastream] Interrupted after {total_bytes} bytes")
                audio_accum.clear()
                session["audio_24k_chunks"] = []
                session["audio_24k_buffer"] = None
                session["audio_24k_total_samples"] = 0
                session["audio_24k_dirty"] = False

                try:
                    await room.local_participant.perform_rpc(
                        destination_identity=agent_identity,
                        method=RPC_PLAYBACK_FINISHED,
                        payload=_json.dumps({
                            "playback_position": 0.0,
                            "interrupted": True,
                        }),
                    )
                except Exception as e:
                    logger.info(f"[datastream] Failed to send playback_finished RPC: {e}")
                continue

            # ── End of segment: normalize to 24k mono, then batch-process ──
            # This is identical to the /generate endpoint which produces correct-speed video.
            raw_bytes = bytes(audio_accum)
            audio_accum.clear()
            if len(raw_bytes) % 2 != 0:
                raw_bytes = raw_bytes[:-1]
            raw_pcm = np.frombuffer(raw_bytes, dtype=np.int16)

            mono_pcm = _to_mono_int16(raw_pcm, sender_channels)
            audio_24k_pcm = _resample_int16(mono_pcm, sender_sr, 24000)
            audio_24k_f = audio_24k_pcm.astype(np.float32) / 32768.0
            audio_16k = resample_poly(audio_24k_f.astype(np.float64), 2, 3).astype(np.float32)
            audio_duration = len(audio_24k_pcm) / 24000.0

            # Store canonical 24k mono PCM for room audio publishing.
            session["audio_24k_chunks"] = [audio_24k_pcm]
            session["audio_24k_buffer"] = audio_24k_pcm
            session["audio_24k_total_samples"] = len(audio_24k_pcm)
            session["audio_24k_dirty"] = False

            num_f = math.ceil(len(audio_16k) / 16000 * DEFAULT_FPS)
            logger.info(
                f"[datastream] Segment end: {total_bytes} bytes -> "
                f"{len(audio_24k_pcm)} samples @24k ({audio_duration:.2f}s), "
                f"expected {num_f} frames"
            )

            # Setup offline pipeline with queue-based writer
            output_queue = queue_mod.Queue(maxsize=100)
            source_info = avatar_cache[session["avatar_id"]]

            await loop.run_in_executor(
                None,
                lambda: sdk.setup_from_cache(
                    source_info, "/dev/null",
                    _output_queue=output_queue,
                    sampling_timesteps=session["sampling_timesteps"],
                    fps=session["fps"],
                ),
            )
            sdk.setup_Nd(N_d=num_f)

            # Signal publisher to enter speech mode
            session["output_queue"] = output_queue
            pipeline_active = True

            # Extract features and run pipeline (all in one batch)
            aud_feat = await loop.run_in_executor(
                None, lambda: sdk.wav2feat.wav2feat(audio_16k, sr=16000),
            )
            logger.info(f"[datastream] Offline wav2feat: {len(aud_feat)} features for {audio_duration:.2f}s audio "
                  f"(expected {num_f}, wav2feat_hardcoded_25fps={int(len(audio_16k)/16000*25)})")

            sdk.audio2motion_queue.put(aud_feat)
            await loop.run_in_executor(None, sdk.close)

            # Wait for unified publisher to finish speech mode
            speech_done_event = session["_speech_done_event"]
            speech_done_event.clear()
            try:
                await asyncio.wait_for(speech_done_event.wait(), timeout=max(audio_duration * 3, 60))
            except asyncio.TimeoutError:
                logger.info(f"[datastream] WARNING: speech_done timeout after {audio_duration*3:.0f}s")

            was_interrupted = session.get("_speech_interrupted", False)
            playback_duration = session.get("audio_24k_total_samples", 0) / 24000.0

            # Reset for next speech segment
            session["audio_24k_chunks"] = []
            session["audio_24k_buffer"] = None
            session["audio_24k_total_samples"] = 0
            session["audio_24k_dirty"] = False
            pipeline_active = False

            # Notify agent that playback finished
            try:
                await room.local_participant.perform_rpc(
                    destination_identity=agent_identity,
                    method=RPC_PLAYBACK_FINISHED,
                    payload=_json.dumps({
                        "playback_position": 0.0 if was_interrupted else playback_duration,
                        "interrupted": was_interrupted,
                    }),
                )
                if was_interrupted:
                    logger.info(f"[datastream] Playback interrupted — sent playback_finished(interrupted=True)")
            except Exception as e:
                logger.info(f"[datastream] Failed to send playback_finished RPC: {e}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.info(f"[datastream] Audio receiver error: {e}")
        import traceback; traceback.print_exc()
    finally:
        # Clean up pipeline if still active
        if pipeline_active:
            try:
                sdk.stop_event.set()
                sdk.close()
            except Exception:
                pass
            session["output_queue"] = None
        logger.info(f"[datastream] Audio receiver stopped for session={session_id}")


@app.post("/start_session")
async def start_session(request: StartSessionRequest):
    """Start a streaming session: Ditto joins LiveKit room and publishes video.

    The agent calls this with room credentials. Ditto connects as a participant,
    creates a video track, and starts publishing frames. Audio arrives via the
    WebSocket /stream endpoint.
    """
    if not LIVEKIT_AVAILABLE:
        raise HTTPException(status_code=500, detail="livekit SDK not installed on this server")

    if request.avatar_id not in avatar_cache:
        raise HTTPException(status_code=400, detail=f"Avatar '{request.avatar_id}' not found. Call /register first.")

    import uuid
    session_id = str(uuid.uuid4())[:8]

    try:
        # Connect to LiveKit room with retry on transient failures
        room = lk_rtc.Room()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await room.connect(request.livekit_url, request.livekit_token)
                break
            except Exception as conn_err:
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt  # exponential backoff: 1s, 2s
                    logger.info(f"[stream] LiveKit connect attempt {attempt + 1} failed: {conn_err}, retrying in {wait_s}s...")
                    await asyncio.sleep(wait_s)
                else:
                    raise
        logger.info(f"[stream] Session {session_id}: connected to LiveKit room '{room.name}', avatar={request.avatar_id}")

        # Create video source matching the registered avatar's resolution.
        # With max_dim=512 registration, the avatar is ~512x340 or similar.
        # No resize needed in publisher since the pipeline outputs at this size.
        avatar_img = avatar_cache[request.avatar_id]["img_rgb_lst"][0]
        pub_h, pub_w = avatar_img.shape[:2]
        video_source = lk_rtc.VideoSource(
            width=pub_w,
            height=pub_h,
        )
        video_track = lk_rtc.LocalVideoTrack.create_video_track(
            "ditto-avatar-video", video_source,
        )
        video_opts = lk_rtc.TrackPublishOptions(
            source=lk_rtc.TrackSource.SOURCE_CAMERA,
            video_codec=lk_rtc.VideoCodec.VP8,
            video_encoding=lk_rtc.VideoEncoding(
                max_bitrate=2_000_000,
                max_framerate=float(request.fps),
            ),
        )
        await room.local_participant.publish_track(video_track, video_opts)
        logger.info(f"[stream] Session {session_id}: video track published")

        # Create audio source and publish track (24kHz mono — matches TTS output)
        audio_source = lk_rtc.AudioSource(sample_rate=24000, num_channels=1, queue_size_ms=100)
        audio_track = lk_rtc.LocalAudioTrack.create_audio_track(
            "ditto-avatar-audio", audio_source,
        )
        audio_opts = lk_rtc.TrackPublishOptions(source=lk_rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(audio_track, audio_opts)
        logger.info(f"[stream] Session {session_id}: audio track published")

        # Create AVSynchronizer — Rust FFI handles FPS pacing + RTP timestamps
        av_sync = lk_rtc.AVSynchronizer(
            audio_source=audio_source,
            video_source=video_source,
            video_fps=request.fps,
            video_queue_size_ms=100,
        )

        with active_sessions_lock:
            active_sessions[session_id] = {
                "room": room,
                "video_source": video_source,
                "video_track": video_track,
                "audio_source": audio_source,
                "audio_track": audio_track,
                "av_sync": av_sync,
                "avatar_id": request.avatar_id,
                "fps": request.fps,
                "sampling_timesteps": request.sampling_timesteps,
                "publish_w": pub_w,
                "publish_h": pub_h,
                "output_queue": None,  # set by audio receiver to trigger speech mode
                "publisher_task": None,
                "audio_recv_task": None,
                # Incremental audio buffer — 24kHz int16 PCM
                "audio_24k_chunks": [],
                "audio_24k_buffer": None,
                "audio_24k_total_samples": 0,
                "audio_24k_dirty": False,
                "_total_frames_published": 0,
                "_speech_done_event": asyncio.Event(),
                "_speech_interrupted": False,
                "_created_at": time.time(),
            }

        # Auto-stop session when agent or room disconnects (handles crashes/abrupt drops)
        agent_identity_for_disconnect = request.agent_identity

        @room.on("participant_disconnected")
        def _on_participant_disconnected(participant):
            if session_id not in active_sessions:
                return
            if participant.identity == agent_identity_for_disconnect:
                logger.info(f"[stream] Agent {participant.identity} disconnected — auto-stopping session {session_id}")
                asyncio.create_task(_auto_stop_session(session_id))

        @room.on("disconnected")
        def _on_room_disconnected(*args):
            if session_id in active_sessions:
                logger.info(f"[stream] Room disconnected — auto-stopping session {session_id}")
                asyncio.create_task(_auto_stop_session(session_id))

        publisher_task = asyncio.create_task(
            _unified_publisher(session_id, active_sessions[session_id])
        )
        active_sessions[session_id]["publisher_task"] = publisher_task
        logger.info(f"[stream] Session {session_id}: unified publisher started")

        # Start DataStream audio receiver if agent identity provided
        if request.agent_identity:
            audio_recv_task = asyncio.create_task(
                _audio_receiver_task(session_id, active_sessions[session_id], request.agent_identity)
            )
            active_sessions[session_id]["audio_recv_task"] = audio_recv_task
            logger.info(f"[stream] Session {session_id}: DataStream audio receiver started (agent={request.agent_identity})")

        return {
            "session_id": session_id,
            "status": "connected",
            "avatar_id": request.avatar_id,
            "room": room.name,
        }

    except Exception as e:
        logger.info(f"[stream] Failed to start session: {e}")
        logger.error(f"[error] Failed to connect to LiveKit room: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to LiveKit room")


SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))  # 30 min default


async def _auto_stop_session(session_id: str):
    """Shared cleanup used by both /stop_session and auto-disconnect handlers."""
    with active_sessions_lock:
        session = active_sessions.pop(session_id, None)
    if session is None:
        return
    try:
        if session.get("audio_recv_task") and not session["audio_recv_task"].done():
            session["audio_recv_task"].cancel()
        if session.get("publisher_task") and not session["publisher_task"].done():
            session["publisher_task"].cancel()
        if session.get("av_sync"):
            await session["av_sync"].aclose()
        await session["room"].disconnect()
        logger.info(f"[stream] Session {session_id}: auto-stopped and room disconnected")
    except Exception as e:
        logger.info(f"[stream] Error in auto-stop for session {session_id}: {e}")


async def _session_cleanup_loop():
    """Periodically check for stale sessions and clean them up."""
    while True:
        await asyncio.sleep(60)  # check every minute
        now = time.time()
        with active_sessions_lock:
            stale = [sid for sid, s in active_sessions.items()
                     if now - s.get("_created_at", now) > SESSION_TTL_SECONDS]
        for sid in stale:
            logger.info(f"[cleanup] Session {sid} exceeded TTL ({SESSION_TTL_SECONDS}s), auto-stopping")
            await _auto_stop_session(sid)


@app.post("/stop_session/{session_id}")
async def stop_session(session_id: str):
    """Stop a streaming session and disconnect from LiveKit room."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    await _auto_stop_session(session_id)
    return {"session_id": session_id, "status": "stopped"}


_DEBUG_ENDPOINTS = os.environ.get("DITTO_DEBUG_ENDPOINTS", "1") == "1"


@app.get("/test_viewer")
async def test_viewer():
    """Serve the bare-bones frame viewer HTML page. Disabled in production (DITTO_DEBUG_ENDPOINTS=0)."""
    if not _DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Debug endpoints disabled")
    html_path = Path(__file__).resolve().parent / "test_viewer.html"
    return HTMLResponse(html_path.read_text())


@app.websocket("/ws/test_frames")
async def ws_test_frames(ws: WebSocket, avatar_id: str = "imogen", duration: float = 5.0):
    """Generate frames from test audio and stream as JPEGs over WebSocket.

    No LiveKit involved — pure Ditto pipeline → JPEG → WebSocket → browser canvas.
    This isolates whether the speed issue is in Ditto or in LiveKit/WebRTC.
    Disabled in production (DITTO_DEBUG_ENDPOINTS=0).
    """
    import json as _json
    import queue as queue_mod

    await ws.accept()
    if not _DEBUG_ENDPOINTS:
        await ws.send_json({"type": "error", "message": "Debug endpoints disabled"})
        await ws.close()
        return

    if avatar_id not in avatar_cache:
        await ws.send_text(_json.dumps({"type": "error", "msg": f"Avatar {avatar_id} not found"}))
        await ws.close()
        return

    sdk = load_ditto()
    source_info = avatar_cache[avatar_id]

    # Generate sine tone test audio at 16kHz
    t = np.linspace(0, duration, int(16000 * duration), dtype=np.float32)
    audio_16k = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine
    num_f = math.ceil(len(audio_16k) / 16000 * DEFAULT_FPS)

    # Also make 24kHz WAV for browser playback
    audio_24k = resample_poly(audio_16k.astype(np.float64), 3, 2).astype(np.float32)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_24k, 24000, format='WAV', subtype='PCM_16')
    wav_bytes = wav_buf.getvalue()

    # Send audio to browser
    await ws.send_text(_json.dumps({
        "type": "audio",
        "data": base64.b64encode(wav_bytes).decode(),
        "duration": duration,
    }))

    # Run offline pipeline with queue writer
    output_queue = queue_mod.Queue(maxsize=200)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: sdk.setup_from_cache(
            source_info, "/dev/null",
            _output_queue=output_queue,
            sampling_timesteps=DEFAULT_SAMPLING_TIMESTEPS,
            fps=DEFAULT_FPS,
        ),
    )
    sdk.setup_Nd(N_d=num_f)

    aud_feat = await loop.run_in_executor(
        None, lambda: sdk.wav2feat.wav2feat(audio_16k, sr=16000),
    )
    logger.info(f"[test_frames] {len(aud_feat)} features for {duration}s audio, expecting {num_f} frames")
    sdk.audio2motion_queue.put(aud_feat)

    # Drain frames in background
    async def drain_and_send():
        frame_count = 0
        while True:
            frame = await loop.run_in_executor(
                None, lambda: output_queue.get(timeout=10),
            )
            if frame is None:
                break
            # Encode as JPEG and send
            _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            await ws.send_bytes(jpeg.tobytes())
            frame_count += 1
        return frame_count

    # Close pipeline (triggers frame generation)
    close_task = loop.run_in_executor(None, sdk.close)
    frame_count = await drain_and_send()
    await close_task

    # Tell browser all frames sent
    await ws.send_text(_json.dumps({
        "type": "frame_count",
        "count": frame_count,
        "fps": DEFAULT_FPS,
        "expected": num_f,
    }))

    logger.info(f"[test_frames] Sent {frame_count} frames to browser")
    await ws.close()


if __name__ == "__main__":
    port = int(os.environ.get("DITTO_PORT", "8181"))
    uvicorn.run(app, host="0.0.0.0", port=port)
