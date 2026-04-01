# DiffPoseTalk — Audio-to-Motion Conversion

## Purpose

Converts speech audio into 3D face motion parameters (FLAME format). This bridges the gap between audio input and video generation — SkyReels-A1 needs driving motion to animate a portrait, and DiffPoseTalk generates that motion from speech. Uses a diffusion model conditioned on audio features.

## Key Files

- `diffposetalk.py` — Main DiffPoseTalk model class. Loads checkpoint, runs audio-to-motion diffusion inference.
- `diff_talking_head.py` — Talking head generation wrapper. Combines audio feature extraction with pose prediction.
- `hubert.py` — HuBERT audio feature extractor for DiffPoseTalk (separate from Ditto's HuBERT).
- `wav2vec2.py` — Wav2Vec2 audio feature extractor (alternative to HuBERT).
- `common.py` — Shared utilities: normalization, tensor ops, motion smoothing.
- `utils/rotation_conversions.py` — Rotation format conversions: Rodrigues, quaternion, rotation matrix, axis-angle (19K).
- `utils/renderer.py` — 3D mesh rendering via pyrender/trimesh for visualization.
- `utils/media.py` — Media file I/O helpers.
- `utils/common.py` — General frame processing and alignment utilities.

## Data Flow

```
Speech Audio (WAV) → hubert.py or wav2vec2.py (audio features)
→ diffposetalk.py (diffusion: audio features → FLAME parameters)
→ FLAME parameters (jaw, expression, head pose) → SkyReels-A1 pipeline
```

## Dependencies

- **Depends on:** DiffPoseTalk checkpoint (`iter_0110000.pt`), stats file (`stats_train.npz`), HuBERT/Wav2Vec2 models
- **Depended on by:** `skyreels/inference_audio.py`, `skyreels/inference_audio_long_video.py`

## Conventions

- FLAME parameters: jaw (3D), expression (50D), head pose (rotation + translation)
- Audio features extracted at model-specific rates, resampled to match video FPS
- Rotation conversions in `utils/rotation_conversions.py` follow PyTorch3D conventions
- Checkpoint loaded via `torch.load()` with key filtering for encoder weights
