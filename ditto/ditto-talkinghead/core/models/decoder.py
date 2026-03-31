"""SPADE Decoder: renders 3D feature volumes into RGB face images.

Takes warped 3D feature volumes from the WarpNetwork and decodes them into
photorealistic face crops using Spatially-Adaptive Denormalization (SPADE).

The decoder outputs a (1, 3, H, W) tensor in [0, 1] range, which is then
transposed to (H, W, 3) and scaled to [0, 255] for compositing.

Supports PyTorch, ONNX, and TensorRT backends.
"""

import numpy as np
import torch
from ..utils.load_model import load_model


class Decoder:
    """SPADE decoder that converts feature volumes to face images."""

    def __init__(self, model_path: str, device: str = "cuda"):
        kwargs = {"module_name": "SPADEDecoder"}
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, warped_features: np.ndarray) -> np.ndarray:
        """Decode warped features into an RGB face image.

        Args:
            warped_features: Warped 3D feature volume from warp network.

        Returns:
            Rendered face image (H, W, 3), float32, range [0, 255].
        """
        if self.model_type == "onnx":
            raw_output = self.model.run(None, {"feature": warped_features})[0]

        elif self.model_type == "tensorrt":
            self.model.setup({"feature": warped_features})
            self.model.infer()
            raw_output = self.model.buffer["output"][0].copy()

        elif self.model_type == "pytorch":
            with torch.no_grad(), torch.autocast(
                device_type=self.device[:4], dtype=torch.float16, enabled=True,
            ):
                raw_output = (
                    self.model(torch.from_numpy(warped_features).to(self.device))
                    .float().cpu().numpy()
                )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Convert from (1, 3, H, W) float [0,1] -> (H, W, 3) float [0,255]
        rendered_image = np.transpose(raw_output[0], [1, 2, 0]).clip(0, 1) * 255
        return rendered_image
