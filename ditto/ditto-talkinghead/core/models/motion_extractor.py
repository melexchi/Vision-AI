"""Motion Extractor: extracts facial motion parameters from a face image.

Takes a cropped, normalized face image (256x256) and predicts the full set
of motion parameters: head pose (pitch/yaw/roll as 66-bin distributions),
face scale, translation, expression coefficients, and canonical keypoints.

This is used during avatar registration to capture the source face's
pose and expression, which become the baseline for animation.

Supports PyTorch, ONNX, and TensorRT backends.
"""

import numpy as np
import torch
from ..utils.load_model import load_model

# The motion extractor outputs these parameters in order.
# Total: 66+66+66+3+63+1+63 = 328 dimensions
OUTPUT_PARAM_NAMES = [
    "pitch",    # (1, 66) — 66-bin head pitch distribution
    "yaw",      # (1, 66) — 66-bin head yaw distribution
    "roll",     # (1, 66) — 66-bin head roll distribution
    "t",        # (1, 3)  — translation (tx, ty, tz)
    "exp",      # (1, 63) — expression coefficients (21 landmarks * 3)
    "scale",    # (1, 1)  — face scale factor
    "kp",       # (1, 63) — canonical keypoints (21 landmarks * 3)
]


class MotionExtractor:
    """Extracts facial motion parameters from a normalized face image."""

    def __init__(self, model_path: str, device: str = "cuda"):
        kwargs = {"module_name": "MotionExtractor"}
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device
        self.output_names = OUTPUT_PARAM_NAMES

    def __call__(self, image: np.ndarray) -> dict:
        """Extract motion parameters from a face image.

        Args:
            image: Normalized face image (1, 3, 256, 256), RGB, float32 [0, 1].

        Returns:
            Dict mapping parameter names to numpy arrays:
                pitch, yaw, roll: (1, 66) — bin distributions
                t: (1, 3) — translation
                exp: (1, 63) — expression (flattened from 21*3)
                scale: (1, 1) — face scale
                kp: (1, 63) — canonical keypoints (flattened from 21*3)
        """
        outputs = {}

        if self.model_type == "onnx":
            raw_outputs = self.model.run(None, {"image": image})
            for idx, name in enumerate(self.output_names):
                outputs[name] = raw_outputs[idx]

        elif self.model_type == "tensorrt":
            self.model.setup({"image": image})
            self.model.infer()
            for name in self.output_names:
                outputs[name] = self.model.buffer[name][0].copy()

        elif self.model_type == "pytorch":
            with torch.no_grad(), torch.autocast(
                device_type=self.device[:4], dtype=torch.float16, enabled=True,
            ):
                raw_outputs = self.model(torch.from_numpy(image).to(self.device))
                for idx, name in enumerate(self.output_names):
                    outputs[name] = raw_outputs[idx].float().cpu().numpy()

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Flatten expression and keypoint tensors to (1, 63) for downstream compatibility
        outputs["exp"] = outputs["exp"].reshape(1, -1)
        outputs["kp"] = outputs["kp"].reshape(1, -1)
        return outputs
