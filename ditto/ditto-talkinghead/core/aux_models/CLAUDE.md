# Auxiliary Models — Face Detection, Landmarks, Audio Encoding

## Purpose

Pre-trained models for face analysis and audio feature extraction. These are not the core generative models — they handle input preprocessing: detecting faces, extracting landmarks, and converting audio to neural features.

## Key Files

- `hubert_stream.py` — Streaming HuBERT audio encoder. Processes audio chunks and extracts 1024-dim features at 50Hz. Supports TensorRT engine.
- `insightface_det.py` — InsightFace face detector. Finds face bounding boxes in source images.
- `insightface_landmark106.py` — 106-point coarse facial landmark extractor.
- `landmark203.py` — 203-point fine facial landmark extractor for detailed motion.
- `mediapipe_landmark478.py` — MediaPipe 478-point face mesh for eye tracking and blendshapes.
- `blaze_face.py` — BlazeFace lightweight face detector (alternative to InsightFace).
- `face_mesh.py` — Face mesh extraction utilities.
- `modules/` — PyTorch implementations of the above models:
  - `hubert_stream.py` — HuBERT model architecture
  - `landmark106.py`, `landmark203.py`, `landmark478.py` — Landmark network architectures
  - `retinaface.py` — RetinaFace detector architecture

## Data Flow

```
Source Image → insightface_det (bbox) → landmark106/203 (keypoints) → source2info
Audio (16kHz) → hubert_stream (HuBERT features) → wav2feat → audio2motion
```

## Dependencies

- **Depends on:** Model weights in `checkpoints/` (ONNX or TensorRT engines), `core/utils/load_model.py`
- **Depended on by:** `core/atomic_components/` (wav2feat, source2info, avatar_registrar)

## Conventions

- Top-level files are wrappers that load and run the model
- `modules/` contains the PyTorch `nn.Module` definitions
- Models loaded via `load_model()` which auto-dispatches by file extension (.engine/.onnx/.pt)
- All models support batch size 1 inference
