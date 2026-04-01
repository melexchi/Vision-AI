# SkyReels — Expression Clip Generation

## Purpose

Generates pre-rendered facial expression clips (idle, thinking, greeting, etc.) using the SkyReels-A1 image-to-video diffusion model (8B parameters, CogVideoX-based). These clips are served by Ditto during idle moments in conversation. Also provides a Gradio UI for interactive portrait animation.

## Key Files

- `app.py` — Gradio web interface for interactive video generation from portrait + driving video
- `inference.py` — Image-to-video inference driven by a reference video
- `inference_audio.py` — Audio-to-video inference driven by speech audio
- `inference_long_video.py` — Long-form video generation with segment stitching
- `inference_audio_long_video.py` — Long-form audio-driven generation
- `setup.sh` — One-time setup: creates venv, installs PyTorch + pytorch3d, downloads SkyReels-A1 (~15GB), DiffPoseTalk, and FILM weights
- `requirements.txt` — Pip dependencies (diffusers, transformers, mediapipe, insightface, etc.)
- `skyreels_a1/` — Main diffusion pipeline (see its own CLAUDE.md)
- `diffposetalk/` — Audio-to-FLAME motion conversion (see its own CLAUDE.md)
- `eval/` — Quality metrics: ArcFace similarity (`arc_score.py`), expression (`expression_score.py`), pose (`pose_score.py`)
- `scripts/demo.py` — Pipeline initialization helpers and inference utility functions
- `assets/` — Example media: 6 driving audio WAVs, 8 driving video MP4s, 20 reference portrait PNGs

## Data Flow

```
Portrait Image + Driving Video → SkyReels-A1 Pipeline → MP4 clip
Portrait Image + Audio → DiffPoseTalk (audio→motion) → SkyReels-A1 → MP4 clip
Ditto subprocess → prerender_clips.py → SkyReels inference → clips saved to CLIPS_DIR
```

## Dependencies

- **Depends on:** HuggingFace models (SkyReels-A1-5B, DiffPoseTalk, FILM), pytorch3d (built from source)
- **Depended on by:** Ditto (`prerender_clips.py` calls this as a subprocess)

## Conventions

- Runs in its own Python venv (`.venv/`), separate from Ditto's uv environment
- Setup tracked by `.setup_done` marker file — delete to force re-setup
- Inference scripts are CLI-based with argparse, not imported as libraries
- Models downloaded to `pretrained_models/` subdirectory

## Common Commands

```bash
bash setup.sh                     # One-time setup (venv + models, ~15GB download)
.venv/bin/python app.py           # Launch Gradio UI
.venv/bin/python inference_audio.py --ref_image <img> --audio <wav> --output <mp4>
```
