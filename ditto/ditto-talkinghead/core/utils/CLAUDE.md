# Core Utilities — TRT Wrapper, Cropping, Blending

## Purpose

Shared utility functions used across the Ditto pipeline. Provides TensorRT engine management, face cropping and alignment, mask generation, and performance-critical Cython blending.

## Key Files

- `tensorrt_utils.py` — TRT engine wrapper: `TRTWrapper` class for loading .engine files, `DynamicShapeOutputAllocator` for variable-size outputs, `check_cuda_errors()` for debugging. Central to all TRT model inference.
- `crop.py` — Face cropping and alignment. Parses landmarks from multiple formats (5/9/68/101/106/203 points), computes affine transforms, `crop_image()` and `paste_back()` for face extraction and composition.
- `load_model.py` — Unified model loader. `load_model(path)` dispatches by extension: `.engine` → TRTWrapper, `.onnx` → ONNX Runtime, `.pt` → PyTorch. Used by all models.
- `eye_info.py` — MediaPipe eye landmark indices (`EyeIdxMP`) and `EyeAttrUtilsByMP` for blink detection and eye openness.
- `get_mask.py` — Face mask generation for blending composited faces back onto original images.
- `blend/` — Cython/C-accelerated blending:
  - `blend.pyx` — Cython wrapper
  - `blend_impl.c`, `blend_impl.h` — C implementation for fast alpha blending
  - Compiled during `setup.sh` via pyximport

## Data Flow

```
load_model.py ← called by all atomic_components and aux_models to load weights
crop.py ← called by source2info (extract face) and putback (paste face back)
tensorrt_utils.py ← called by every TRT model for engine execution
blend/ ← called by putback for final frame compositing
```

## Dependencies

- **Depends on:** TensorRT, ONNX Runtime, PyTorch (runtime backends), NumPy, OpenCV
- **Depended on by:** Everything in `core/` — this is the foundational utility layer

## Conventions

- `load_model()` is the single entry point for all model loading — never load directly
- TRTWrapper handles dynamic shapes via output allocator callback
- `crop.py` supports 6 different landmark formats — use `parse_pt2_from_*()` helpers
- Cython blend is optional — falls back to NumPy if compilation fails
