# Atomic Components — Pipeline Stages

## Purpose

The 12 modular stages of the Ditto lip-sync pipeline. Each component is a self-contained class that performs one step of the audio-to-video transformation. They are chained together via queues in `stream_pipeline_offline.py`.

## Key Files

- `wav2feat.py` — Converts 16kHz audio → HuBERT features (1024-dim at 50Hz). Wraps `HubertStreaming` from `aux_models/`.
- `audio2motion.py` — HuBERT features → motion parameters via LMDM diffusion (5 DDIM steps). The bottleneck stage.
- `motion_stitch.py` — Blends predicted motion with source avatar keypoints for natural movement.
- `warp_f3d.py` — Warps 3D face mesh using predicted motion via GridSample3D (requires TRT plugin).
- `decode_f3d.py` — Decodes warped features into RGB face frames.
- `putback.py` — Composites the generated face back onto the original image with blending.
- `avatar_registrar.py` — Registers avatars: runs face detection + landmark extraction + feature caching.
- `source2info.py` — Extracts per-frame features (keypoints, appearance, motion) from source image/video.
- `loader.py` — Loads source images/videos into RGB frame lists.
- `condition_handler.py` — Builds conditioning vectors (emotion, eye state, head shape).
- `writer.py` — Writes output frames as MP4 (via imageio) or pushes to queue for streaming.
- `cfg.py` — Pipeline configuration constants and defaults.

## Data Flow

```
loader → source2info → avatar_registrar (registration path)
wav2feat → audio2motion → motion_stitch → warp_f3d → decode_f3d → putback → writer (generation path)
```

## Dependencies

- **Depends on:** `core/aux_models/` (HuBERT, InsightFace, MediaPipe), `core/models/` (LMDM, decoder, warp), `core/utils/` (crop, blend, TRT wrapper)
- **Depended on by:** `stream_pipeline_offline.py`, `stream_pipeline_online.py`

## Conventions

- Each component is a class with `__init__(cfg: dict)` and `__call__(...)` interface
- Config dicts use `_cfg` suffix: `w2f_cfg`, `lmdm_cfg`, `stitch_cfg`
- Relative imports: `from ..aux_models.hubert_stream import HubertStreaming`
- NumPy arrays for image data, PyTorch tensors for model I/O
