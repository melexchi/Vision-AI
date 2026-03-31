"""Decode Feature 3D: renders a 3D feature volume into an RGB image.

Uses the SPADE (Spatially-Adaptive Denormalization) decoder to convert the
warped 3D feature volume back into a visible face image. The decoder handles
the neural rendering step, producing a photorealistic 512x512 crop from the
abstract feature representation.
"""

from ..models.decoder import Decoder


class DecodeF3D:
    """Thin wrapper around the SPADE decoder model.

    The decoder takes a warped 3D feature volume and produces an RGB image crop.
    This is the final GPU-intensive step before compositing onto the background.
    """

    def __init__(self, decoder_cfg: dict):
        self.decoder = Decoder(**decoder_cfg)

    def __call__(self, warped_features):
        """Decode warped 3D features into an RGB face image.

        Args:
            warped_features: Warped feature volume from WarpF3D.

        Returns:
            Rendered face image (H, W, 3), float32, range 0-255.
        """
        return self.decoder(warped_features)
