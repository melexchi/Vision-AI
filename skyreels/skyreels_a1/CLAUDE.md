# SkyReels-A1 — Video Diffusion Pipeline

## Purpose

The core image-to-video diffusion pipeline based on CogVideoX architecture. Takes a portrait image and driving motion (from video or audio-derived FLAME parameters) and generates animated video using an 8B-parameter 3D transformer with DDIM sampling.

## Key Files

- `skyreels_a1_i2v_pipeline.py` — Main image-to-video pipeline. Extends HuggingFace `DiffusionPipeline` with pose-conditioned generation, T5 text encoding, and DDIM denoising loop.
- `skyreels_a1_i2v_long_pipeline.py` — Extended pipeline for long video generation with segment overlap and temporal consistency.
- `models/transformer3d.py` — 3D transformer architecture (36K, largest model file). Processes spatiotemporal video tokens with attention and feed-forward blocks.
- `ddim_solver.py` — DDIM diffusion solver for deterministic sampling with configurable step count.
- `pre_process_lmk3d.py` — Preprocesses 3D face landmarks from driving video into pose conditioning tensors.
- `pipeline_output.py` — Output container class for pipeline results.
- `src/frame_interpolation.py` — FILM-based frame interpolation for smooth transitions between generated segments.
- `src/smirk_encoder.py` — SMIRK expression encoder for extracting facial expression parameters.
- `src/renderer.py` — 3D face mesh renderer using pyrender/trimesh.
- `src/FLAME/` — Basel parametric 3D face model (FLAME.py + linear blend skinning).
- `src/media_pipe/` — MediaPipe face detection and landmark extraction wrappers.
- `src/lmk3d_test.py` — Manual test script for face animation preprocessing.

## Data Flow

```
Portrait + Driving Motion → pre_process_lmk3d (pose conditioning)
T5 Encoder (text prompt) → text embeddings
Pose conditioning + text embeddings + noise → transformer3d (denoising, N DDIM steps)
→ decoded video frames → frame_interpolation (optional upsampling) → MP4
```

## Dependencies

- **Depends on:** HuggingFace diffusers, transformers (T5), FLAME model weights, SMIRK encoder, FILM model
- **Depended on by:** `skyreels/inference*.py` scripts, `skyreels/app.py`, Ditto's `prerender_clips.py`

## Conventions

- Pipelines extend `DiffusionPipeline` from HuggingFace diffusers
- Models downloaded to `pretrained_models/` (symlinked by `setup.sh`)
- `src/` contains auxiliary processors; `models/` contains the main transformer
- Inference uses `torch.no_grad()` and `torch.cuda.amp.autocast()` for efficiency
