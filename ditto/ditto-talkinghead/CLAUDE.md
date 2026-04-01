# Ditto TalkingHead — Core Inference Engine

## Purpose

The inference engine that powers Ditto's lip-sync generation. It provides a multi-stage pipeline that converts audio into animated face video frames, supporting both TensorRT and ONNX Runtime backends. The pipeline runs as multi-threaded workers connected by queues.

## Key Files

- `stream_pipeline_offline.py` — Main batch pipeline: spawns 4 worker threads (audio2motion, gpu_worker, putback, writer) connected by `queue.Queue`, processes complete audio files
- `stream_pipeline_online.py` — Real-time streaming pipeline variant for LiveKit sessions
- `inference.py` — Standalone CLI entry point for offline inference
- `core/atomic_components/` — 12 modular pipeline stages (see its own CLAUDE.md)
- `core/models/` — Neural network model definitions (LMDM, decoder, warp, etc.)
- `core/aux_models/` — Face detection, landmark extraction, HuBERT audio encoder
- `core/utils/` — TRT wrapper, face cropping, blending, model loader
- `scripts/cvt_onnx_to_trt.py` — Converts ONNX models to TensorRT engines with GridSample3D plugin support
- `environment.yaml` — Legacy Conda environment spec (project now uses uv)

## Data Flow

```
Audio (16kHz) → wav2feat (HuBERT) → audio2motion (LMDM) → motion_stitch
→ warp_f3d (GridSample3D) → decode_f3d → putback → writer (MP4 or queue)
```

Pipeline stages run as threads pulling from queues with `threading.Event()` for stop signaling.

## Dependencies

- **Depends on:** Model weights in `checkpoints/` (downloaded by `download_weights.sh`)
- **Depended on by:** `ditto_api.py` (imports via `sys.path.insert`)

## Conventions

- Pipeline stages are classes in `core/atomic_components/` with `__call__` for invocation
- Model loading dispatches by file extension: `.engine` → TRT, `.onnx` → ONNX Runtime, `.pt` → PyTorch
- Config passed as dicts with `_cfg` suffix (e.g., `hubert_cfg`, `lmdm_cfg`)
- Worker threads use `queue.Queue(maxsize=N)` for backpressure

## Common Commands

```bash
uv run python scripts/cvt_onnx_to_trt.py --onnx-dir checkpoints/ditto_onnx --trt-dir checkpoints/ditto_trt
uv run python inference.py --source <image> --audio <wav> --output <mp4>
```
