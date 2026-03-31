# Avatar — Ditto GPU Inference Server

Real-time talking head generation from audio. Runs on RunPod (GPU), serves video to the LiveKit agent on Hetzner.

## Architecture

```
User → LiveKit Agent (Hetzner) → POST /generate → Ditto API (RunPod GPU) → MP4 video
                                  POST /register → caches avatar features (~9MB)
                                  GET  /clips     → pre-rendered idle/thinking/greeting loops
```

## Performance (A100 80GB)

| Metric | Value |
|--------|-------|
| Streaming throughput | 25–50 it/s |
| GPU utilization | ~30% |
| First frame latency | ~2s |
| Sampling timesteps | 5 (DDIM) |
| Output resolution | 512px |

## Pipeline

```
Audio (16kHz) → HuBERT (TRT) → LMDM Diffusion (TRT, 5 steps)
    → Motion Stitch (TRT) → Warp Network (TRT + GridSample3D plugin)
    → Decoder (TRT) → Putback → RGB frames → LiveKit video track
```

All models run in TensorRT FP16. The **GridSample3D plugin** is critical — without it, warp_network falls back to PyTorch and throughput drops to ~13 it/s.

---

## Deployment on RunPod

### Step 1: Install TensorRT + CUDA Python

```bash
pip install --break-system-packages tensorrt-cu12 cuda-python<13
```

Verify:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
python3 -c "from cuda import cuda; print('cuda-python OK')"
```

### Step 2: Build GridSample3D TRT Plugin

```bash
# Clone plugin source
cd /workspace
git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git

# Download TRT headers (pip packages don't include them)
mkdir -p /tmp/trt_headers && cd /tmp/trt_headers
for f in NvInfer.h NvInferRuntime.h NvInferRuntimeBase.h NvInferRuntimePlugin.h \
         NvInferVersion.h NvInferImpl.h NvInferLegacyDims.h NvInferConsistency.h \
         NvInferPlugin.h NvInferRuntimeCommon.h NvInferSafeRuntime.h \
         NvInferConsistencyImpl.h NvInferPluginUtils.h; do
    wget -q "https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.0/include/$f" -O "$f"
done

# Create linker symlink (required — pip only installs libnvinfer.so.10)
TRT_LIBS=/usr/local/lib/python3.12/dist-packages/tensorrt_libs
ln -sf "$TRT_LIBS/libnvinfer.so.10" "$TRT_LIBS/libnvinfer.so"

# Patch CMakeLists to find TRT libs
cd /workspace/grid-sample3d-trt-plugin
sed -i 's|target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer)|target_link_directories(${PROJECT_NAME} PRIVATE ${TensorRT_LIB_DIR})\ntarget_link_libraries(${PROJECT_NAME} PRIVATE nvinfer)|' CMakeLists.txt

# Build
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p build && cd build
cmake .. \
    -DTensorRT_INCLUDE_DIR=/tmp/trt_headers \
    -DTensorRT_LIB_DIR=$TRT_LIBS \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_FLAGS="-I/tmp/trt_headers" \
    -DCMAKE_CUDA_FLAGS="-I/tmp/trt_headers"
make -j4

# Install
cp libgrid_sample_3d_plugin.so /workspace/ditto/checkpoints/ditto_onnx/
```

### Step 3: Build TRT Engines

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:$LD_LIBRARY_PATH

cd /workspace/ditto/ditto-talkinghead

# Delete old engines (they're tied to specific TRT versions)
rm -f checkpoints/ditto_trt_v10/*.engine

# Rebuild all (takes ~10-15 min)
python3 scripts/cvt_onnx_to_trt.py \
    --onnx_dir /workspace/ditto/checkpoints/ditto_onnx \
    --trt_dir checkpoints/ditto_trt_v10 \
    --force-fp16
```

The script auto-detects `libgrid_sample_3d_plugin.so` in the onnx dir and uses it for warp_network.

### Step 4: Update Config PKL

```python
python3 << 'EOF'
import pickle

pkl = "/workspace/ditto/ditto-talkinghead/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_v10.pkl"

with open(pkl, "rb") as f:
    cfg = pickle.load(f)

cfg["base_cfg"]["warp_network_cfg"] = {
    "model_path": "warp_network_fp16.engine",
    "device": "cuda",
    "plugin_path": "/workspace/ditto/checkpoints/ditto_onnx/libgrid_sample_3d_plugin.so",
}
cfg["base_cfg"]["hubert_cfg"] = {
    "model_path": "hubert_fp16.engine",
    "device": "cuda",
}

with open(pkl, "wb") as f:
    pickle.dump(cfg, f)

print("Config updated")
EOF
```

### Step 5: Patch Source for Plugin Loading

Two files need patching so the GridSample3D plugin .so gets loaded at runtime:

**`ditto-talkinghead/core/utils/load_model.py`** — in the TRT engine branch:
```python
elif model_path.endswith(".engine") or model_path.endswith(".trt"):
    from .tensorrt_utils import TRTWrapper
    plugin_file_list = []
    plugin_path = kwargs.get("plugin_path", "")
    if plugin_path:
        plugin_file_list.append(plugin_path)
    model = TRTWrapper(model_path, plugin_file_list=plugin_file_list)
    return model, "tensorrt"
```

**`ditto-talkinghead/core/models/warp_network.py`** — accept and forward `plugin_path`:
```python
def __init__(self, model_path, device="cuda", plugin_path=""):
    kwargs = {"module_name": "WarpingNetwork"}
    if plugin_path:
        kwargs["plugin_path"] = plugin_path
    self.model, self.model_type = load_model(model_path, device=device, **kwargs)
```

### Step 6: Start the Server

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:/workspace/ditto/checkpoints/ditto_onnx:$LD_LIBRARY_PATH

cd /workspace/ditto
DITTO_BACKEND=trt python3 ditto_api.py
```

### Verify

```bash
curl -s localhost:8181/health | python3 -m json.tool
```

Expected:
```json
{
    "status": "ok",
    "sampling_timesteps": 5,
    "fps": 25
}
```

Warmup logs should show **20+ it/s**, streaming should hit **25-50 it/s**.

---

## Key Settings (in `ditto_api.py`)

```python
DEFAULT_SAMPLING_TIMESTEPS = 5   # DDIM steps (lower = faster, 5 ≈ quality of 10)
DEFAULT_FPS = 25                 # Target frame rate
max_size = 512                   # Output resolution (lower = faster)
```

## Directory Structure

```
inference/ditto/
├── ditto-talkinghead/     # Core Ditto engine (models, pipeline, utils)
├── ditto_api.py           # FastAPI server (port 8181)
├── prerender_clips.py     # SkyReels-A1 idle clip generator
├── Dockerfile             # RunPod container definition
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DITTO_PATH` | `./ditto-talkinghead` | Path to Ditto engine directory |
| `DITTO_BACKEND` | `onnx` | Backend: `onnx` (~15 it/s) or `trt` (~25-50 it/s) |
| `AVATAR_CACHE_DIR` | `/workspace/avatar_cache` | Persistent avatar feature cache |
| `AVATAR_CLIPS_DIR` | `/workspace/avatar_clips` | Pre-rendered clip storage |
| `AVATAR_IMAGES_DIR` | `/workspace/avatar_images` | Stored avatar source images |
| `TTS_URL` | `http://localhost:8282/tts/stream` | Orpheus TTS server endpoint |

## Model Checkpoints

```
/workspace/ditto/ditto-talkinghead/checkpoints/
├── ditto_cfg/
│   └── v0.4_hubert_cfg_trt_v10.pkl    # TensorRT config (includes plugin_path)
├── ditto_trt_v10/                      # TensorRT engines
│   ├── warp_network_fp16.engine        # Requires GridSample3D plugin
│   ├── hubert_fp16.engine
│   ├── decoder_fp16.engine
│   ├── stitch_network_fp16.engine
│   ├── appearance_extractor_fp16.engine
│   ├── motion_extractor_fp32.engine
│   └── ...
└── ditto_onnx/
    ├── libgrid_sample_3d_plugin.so     # GridSample3D TRT plugin
    └── *.onnx                          # ONNX models (source for engine builds)
```

## Video Sync Debugging Log

### Problem
Ditto lip-sync video appears to play at ~2x speed on the client. Audio plays at normal speed. Video lip movements finish first, idle resumes, audio continues.

### Key Finding
**Burned-in elapsed-time overlay confirms server-side pacing is CORRECT.**
Timer on video matches user's real clock. Frames are delivered at 20fps, total duration matches audio. The "fast" appearance is in the lip-sync motion content itself, not the delivery rate.

### What We Tried

| # | Change | Location | Result |
|---|--------|----------|--------|
| 1 | AVSynchronizer with 5000ms queue | ditto_api.py | Frames burst through |
| 2 | AVSynchronizer with 100ms queue (official pattern) | ditto_api.py | Better, still fast |
| 3 | Session-level AVSync (one per session, shared idle+speech) | ditto_api.py | Same result |
| 4 | `av_sync.reset()` on idle→speech transition | ditto_api.py | No change |
| 5 | Manual pacing (`asyncio.sleep(1/fps)` + direct `capture_frame`) | ditto_api.py | Worse — user said video quality degraded |
| 6 | `AudioSource(queue_size_ms=100)` (was default 1000ms) | ditto_api.py | Slightly better |
| 7 | Agent-side 50ms chunks (was 405ms) | ditto_avatar.py | No visible change |
| 8 | Dedicated WS receive loop for control messages | ditto_avatar.py | No visible change |
| 9 | D0 pre-seeding (10 silence chunks, discard 70 frames) | ditto_api.py | Fixed 3-4s freeze, didn't help speed |
| 10 | Audio-gated video publishing | ditto_api.py | Frame counts now match `audio_duration * fps` perfectly |
| 11 | Cancel idle on first real frame (not on pipeline setup) | ditto_api.py | Fixed freeze during D0 |
| 12 | Frame duplication (FRAME_REPEAT=2) | ditto_api.py | User says still same speed |
| 13 | Burned-in frame counter + elapsed time | ditto_api.py | **Timer matches real clock** |

### Server-Side Metrics (all correct)
```
[publish] Exiting after 162 frames in 8.1s (20.0fps actual)
[publish] Exiting after 130 frames in 6.5s (20.0fps actual)
[publish] Exiting after 90 frames in 4.5s (20.0fps actual)
```

### Remaining Hypotheses

| Hypothesis | Description | Test |
|-----------|-------------|------|
| A. LMDM motion speed | Model generates exaggerated/fast lip movements intrinsically | Try FRAME_REPEAT=5+ or change model temporal params |
| B. Feature rate mismatch | HuBERT features at 25fps (40ms) played at 20fps (50ms) — 25% stretch may not be enough | Change pipeline FPS to 25 to match feature rate |
| C. VP8 encoder behavior | Software VP8 on RunPod may produce variable frame timing | Try different codec or encoding settings |
| D. WebRTC SFU batching | LiveKit SFU may batch/reorder video frames despite correct server timing | Test on local network |
| E. Idle contrast | 15fps idle → 20fps lip-sync creates perceptual contrast | Match idle FPS to speech FPS |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No module named 'cuda'` | cuda-python missing | `pip install --break-system-packages cuda-python<13` |
| `No module named 'tensorrt'` | tensorrt missing | `pip install --break-system-packages tensorrt-cu12` |
| Engine serialization mismatch | Engines from different TRT version | Delete `.engine` files, rebuild (Step 3) |
| `GridSample3D` plugin not found | Plugin .so missing or wrong TRT version | Rebuild plugin (Step 2) |
| ~13 it/s instead of 25+ | warp_network in PyTorch mode | Check config PKL uses `.engine` not `.pth`, verify plugin_path |
| Video stuttering | FPS > pipeline throughput | Check `streaming: Nit [time, XX.XXit/s]` in logs |

## API Endpoints

### `GET /health`
Health check. Returns cached avatar count and settings.

### `POST /register`
Register an avatar image. Caches pre-computed features for instant reuse.

### `POST /start_session`
Start a real-time streaming session with LiveKit integration.

### `WebSocket /stream`
Real-time audio streaming for lip-sync. Protocol:
1. Send `{"type": "init", "session_id": "..."}`
2. Receive `{"type": "ready"}`
3. Send binary audio chunks (16kHz int16 PCM)
4. Send `{"type": "end"}` to flush
5. Receive `{"type": "done", "frames_generated": N}`

### `POST /generate`
Batch: generate video from audio (returns MP4).

### `GET /avatars`
List cached avatars with readiness status.

### `GET /clips/{avatar_id}/{clip_type}`
Serve pre-rendered clip (idle, thinking, greeting).
