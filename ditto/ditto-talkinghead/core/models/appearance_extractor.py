"""Appearance Feature Extractor: extracts 3D appearance features from a face image.

Takes a cropped, normalized face image (256x256) and produces a 3D feature
volume (1, 32, 16, 64, 64) encoding the face's visual appearance. This feature
volume is later warped by the warp network to match the driving motion.

Supports PyTorch, ONNX, and TensorRT backends.
"""

import numpy as np
import torch
from ..utils.load_model import load_model


class AppearanceExtractor:
    """Extracts a 3D appearance feature volume from a normalized face image."""

    def __init__(self, model_path: str, device: str = "cuda"):
        kwargs = {"module_name": "AppearanceFeatureExtractor"}
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Extract appearance features from a face image.

        Args:
            image: Normalized face image (1, 3, 256, 256), RGB, float32 [0, 1].

        Returns:
            3D feature volume as numpy array (1, 32, 16, 64, 64).
        """
        if self.model_type == "onnx":
            features = self.model.run(None, {"image": image})[0]

        elif self.model_type == "tensorrt":
            self.model.setup({"image": image})
            self.model.infer()
            features = self.model.buffer["pred"][0].copy()

        elif self.model_type == "pytorch":
            with torch.no_grad(), torch.autocast(
                device_type=self.device[:4], dtype=torch.float16, enabled=True,
            ):
                features = (
                    self.model(torch.from_numpy(image).to(self.device))
                    .float()
                    .cpu()
                    .numpy()
                )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return features
