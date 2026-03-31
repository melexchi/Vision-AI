# Avatar AI — Inference Stack

Self-hosted GPU inference stack for real-time AI talking avatars. Combines live lip-sync streaming, pre-rendered expression clips, neural TTS, and camera vision into a cohesive pipeline.

---

## Services

### 🎭 Ditto — Live Lip-Sync Streaming
> Based on [antgroup-creative/ditto-talkinghead](https://github.com/antgroup-creative/ditto-talkinghead)

Real-time talking-head generation driven by audio input. Streams portrait video at 25fps via LiveKit WebRTC.

- **Backend:** TensorRT (Ampere+) for production, PyTorch fallback
- **Resolution:** 340×512 portrait
- **Latency:** ~200ms mouth-to-video with TRT + GridSample3D plugin
- **API:** FastAPI + WebSocket at `:8181`
- **Features:** Avatar registration, pre-rendered idle/expression clips, DDIM sampling (5 steps), VP8 video encoding

```bash
cd ditto
LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/tensorrt_libs:$LD_LIBRARY_PATH \
DITTO_BACKEND=trt uv run ditto_api.py
```

---

### 🎬 SkyReels — Expression Clip Generation
> Based on [SkyworkAI/SkyReels-A1](https://github.com/SkyworkAI/SkyReels-A1)

Generates pre-rendered avatar clips (idle, thinking, greeting, expressions) using the SkyReels-A1 image-to-video diffusion model. Clips are loaded by Ditto and played back during silence/waiting states.

- **Model:** SkyReels-A1 (CogVideoX-based, ~8B params)
- **Input:** Reference image + driving audio/video
- **Output:** MP4 clips at avatar resolution
- **Use case:** Idle animations, thinking loops, greetings

```bash
cd skyreels
python inference_audio.py --ref_image /path/to/avatar.jpg --audio /path/to/audio.wav
```

---

### 🔊 Chatterbox — Text-to-Speech
> Based on [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)

Neural TTS server with voice cloning support. Used as a fallback or standalone TTS when Google Chirp3 HD is not available.

- **Model:** Chatterbox TTS
- **API:** FastAPI REST at `:8080`

```bash
cd chatterbox
pip install -r requirements.txt
python api_server.py
```

---

### 👁️ SmolVLM — Camera Vision
> Based on [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)

Lightweight vision-language model for processing user camera frames. Enables the avatar agent to see and respond to visual context.

- **Model:** SmolVLM2-2.2B-Instruct
- **Inference:** ~1.3s per frame
- **API:** FastAPI REST at `:8282`

```bash
cd smolvlm
uv run smolvlm_server.py
```

---

## Architecture

```
User Browser
    │  WebRTC (LiveKit)
    ▼
LiveKit Agent Server
    │  HTTP/WebSocket
    ├──▶ Ditto API (:8181)  ──▶  TRT Engines  ──▶  VP8 video stream
    │         └── idle clips (SkyReels pre-rendered)
    │
    ├──▶ SmolVLM (:8282)    ──▶  camera frame analysis
    │
    └──▶ Google Chirp3 HD   ──▶  TTS audio
```

---

## Hardware Requirements

| Service | Min GPU | Recommended |
|---------|---------|-------------|
| Ditto (TRT) | A100 40GB | H100 / A100 80GB |
| Ditto (PyTorch) | RTX 3090 | A100 |
| SkyReels | A100 40GB | H100 NVL |
| SmolVLM | RTX 3080 | Any 16GB+ |
| Chatterbox | RTX 3060 | Any 8GB+ |

CUDA 12.x and TensorRT 10.x required for Ditto TRT mode.

---

## Setup

Each service has its own `setup.sh` or `requirements.txt`. See the individual service directories for detailed setup instructions.

**Ditto quick setup:**
```bash
cd ditto
bash setup.sh          # installs deps, downloads weights
bash download_weights.sh
```

**TRT engine build** (Ditto, one-time):
```bash
cd ditto/ditto-talkinghead
python scripts/cvt_onnx_to_trt.py
```
