"""Stitching Network: adjusts keypoints to fix face-background boundary artifacts.

When the face is animated and composited back onto the static background,
there can be visible seams at the boundary. The stitching network is a small
MLP that takes source and driving keypoints and produces corrected driving
keypoints that minimize these boundary artifacts.

Supports PyTorch, ONNX, and TensorRT backends.
"""

import numpy as np
import torch
from ..utils.load_model import load_model


class StitchNetwork:
    """MLP that corrects keypoints to reduce stitching artifacts."""

    def __init__(self, model_path: str, device: str = "cuda"):
        kwargs = {"module_name": "StitchingNetwork"}
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(
        self, source_keypoints: np.ndarray, driving_keypoints: np.ndarray
    ) -> np.ndarray:
        """Correct driving keypoints to reduce boundary artifacts.

        Args:
            source_keypoints: Source face keypoints (1, 21, 3).
            driving_keypoints: Driving face keypoints (1, 21, 3).

        Returns:
            Corrected driving keypoints (1, 21, 3).
        """
        if self.model_type == "onnx":
            output = self.model.run(None, {
                "kp_source": source_keypoints,
                "kp_driving": driving_keypoints,
            })[0]

        elif self.model_type == "tensorrt":
            self.model.setup({
                "kp_source": source_keypoints,
                "kp_driving": driving_keypoints,
            })
            self.model.infer()
            output = self.model.buffer["out"][0].copy()

        elif self.model_type == "pytorch":
            with torch.no_grad():
                output = self.model(
                    torch.from_numpy(source_keypoints).to(self.device),
                    torch.from_numpy(driving_keypoints).to(self.device),
                ).cpu().numpy()

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return output
