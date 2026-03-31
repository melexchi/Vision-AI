"""Warp Network: deforms 3D appearance features using keypoint motion.

Takes the source avatar's 3D feature volume (from the appearance extractor)
and warps it according to the displacement between source and driving keypoints.
This produces a motion-aware feature volume that, when decoded, shows the face
in the target pose and expression.

The warp operation uses 5D grid sampling (3D spatial + batch + channel),
which requires a custom TensorRT plugin (GridSample3D) for GPU acceleration.
Without the plugin, TRT falls back to slower implementations.

Supports PyTorch, ONNX, and TensorRT backends.
"""

import numpy as np
import torch
from ..utils.load_model import load_model


class WarpNetwork:
    """Warps 3D appearance features according to keypoint displacement.

    For single-image avatars, the source features (feature_3d) and source
    keypoints (kp_source) are constant. The cache_source() method stores
    them on GPU to avoid redundant host-to-device copies every frame.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        kwargs = {"module_name": "WarpingNetwork"}
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        # GPU tensor cache for constant-per-avatar inputs
        self._cached_source_features = None    # torch.Tensor on device
        self._cached_source_keypoints = None   # torch.Tensor on device
        self._driving_keypoint_buffer = None   # Pre-allocated torch.Tensor on device
        self._output_buffer = None             # Pre-allocated numpy output buffer
        self._torch_compiled = False

    def cache_source(self, feature_3d: np.ndarray, kp_source: np.ndarray | None = None):
        """Cache constant source tensors on GPU to skip per-frame H2D copies.

        Args:
            feature_3d: Source appearance features (1, 32, 16, 64, 64).
            kp_source: Source keypoints (1, 21, 3), or None if not yet available.
        """
        if self.model_type != "pytorch":
            return

        self._cached_source_features = torch.from_numpy(
            np.ascontiguousarray(feature_3d)
        ).to(self.device)

        if kp_source is not None:
            self._cached_source_keypoints = torch.from_numpy(
                np.ascontiguousarray(kp_source)
            ).to(self.device)

        # Pre-allocate driving keypoint buffer
        keypoint_shape = (1, 21, 3) if kp_source is None else kp_source.shape
        self._driving_keypoint_buffer = torch.empty(
            keypoint_shape, dtype=torch.float32, device=self.device,
        )

        # Apply torch.compile for kernel fusion
        if not self._torch_compiled:
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                self._torch_compiled = True
            except Exception:
                pass

    def clear_cache(self):
        """Release cached GPU tensors."""
        self._cached_source_features = None
        self._cached_source_keypoints = None
        self._driving_keypoint_buffer = None
        self._output_buffer = None

    def __call__(
        self,
        feature_3d: np.ndarray,
        kp_source: np.ndarray,
        kp_driving: np.ndarray,
    ) -> np.ndarray:
        """Warp source features according to keypoint displacement.

        Args:
            feature_3d: Source 3D appearance features (1, 32, 16, 64, 64).
            kp_source: Source keypoints (1, 21, 3).
            kp_driving: Driving (target) keypoints (1, 21, 3).

        Returns:
            Warped 3D feature volume, same shape as feature_3d.
        """
        if self.model_type == "onnx":
            outputs = self.model.run(None, {
                "feature_3d": feature_3d,
                "kp_source": kp_source,
                "kp_driving": kp_driving,
            })
            return outputs[0]

        elif self.model_type == "tensorrt":
            self.model.setup({
                "feature_3d": feature_3d,
                "kp_source": kp_source,
                "kp_driving": kp_driving,
            })
            self.model.infer()
            return self.model.buffer["out"][0].copy()

        elif self.model_type == "pytorch":
            # Use cached GPU tensors when available to skip H2D copies
            source_feat = (
                self._cached_source_features
                if self._cached_source_features is not None
                else torch.from_numpy(feature_3d).to(self.device)
            )
            source_kp = (
                self._cached_source_keypoints
                if self._cached_source_keypoints is not None
                else torch.from_numpy(kp_source).to(self.device)
            )

            # Update driving keypoints in-place if buffer exists
            if self._driving_keypoint_buffer is not None:
                self._driving_keypoint_buffer.copy_(
                    torch.from_numpy(kp_driving), non_blocking=True,
                )
                driving_kp = self._driving_keypoint_buffer
            else:
                driving_kp = torch.from_numpy(kp_driving).to(self.device)

            with torch.no_grad(), torch.autocast(
                device_type=self.device[:4], dtype=torch.float16, enabled=True,
            ):
                output = self.model(source_feat, source_kp, driving_kp)
                return output.float().cpu().numpy()

        raise ValueError(f"Unsupported model type: {self.model_type}")
