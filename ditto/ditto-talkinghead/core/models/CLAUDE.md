# Core Models — Neural Network Architectures

## Purpose

The generative neural networks that power Ditto's lip-sync pipeline. These models convert audio features into motion parameters and render animated face frames. All support TensorRT acceleration.

## Key Files

- `lmdm.py` — Latent Motion Diffusion Model. Converts HuBERT audio features to face motion parameters via 5-step DDIM diffusion. The core generation model.
- `decoder.py` — Final video frame decoder. Converts warped feature maps into RGB pixel output.
- `warp_network.py` — Warps 3D face features using predicted motion via GridSample3D. Requires custom TRT plugin.
- `appearance_extractor.py` — Extracts appearance features from the source avatar image.
- `motion_extractor.py` — Extracts motion parameters from source for motion delta computation.
- `stitch_network.py` — Blends predicted motion with source keypoints for seamless transitions.
- `modules/` — Sub-modules and building blocks:
  - `LMDM.py` — LMDM transformer architecture
  - `lmdm_modules/` — Rotary embeddings, transformer blocks, utilities
  - `convnextv2.py` — ConvNextV2 backbone
  - `dense_motion.py` — Dense optical flow estimation
  - `spade_generator.py` — SPADE conditional generator
  - `warping_network.py` — Warping implementation
  - `appearance_feature_extractor.py` — Appearance feature CNN
  - `stitching_network.py` — Stitching MLP

## Data Flow

```
appearance_extractor(source_image) → appearance features
motion_extractor(source_image) → source motion
lmdm(audio_features) → predicted motion delta
stitch_network(source_motion + delta) → final motion
warp_network(appearance, motion) → warped features
decoder(warped features) → RGB frames
```

## Dependencies

- **Depends on:** Model weights (PyTorch .pt, ONNX .onnx, or TensorRT .engine), `core/utils/tensorrt_utils.py`
- **Depended on by:** `core/atomic_components/` (each component wraps one of these models)

## Conventions

- Top-level files are inference wrappers, `modules/` has `nn.Module` definitions
- TensorRT engines use FP16 by default, except LMDM and HuBERT (FP32 for accuracy)
- GridSample3D requires `libgrid_sample_3d_plugin.so` loaded at TRT build time
- Model precision configured in `scripts/cvt_onnx_to_trt.py`
