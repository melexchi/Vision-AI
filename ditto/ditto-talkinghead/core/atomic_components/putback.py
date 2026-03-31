"""Putback: composites the rendered face back onto the original frame.

After the pipeline renders a cropped face region, this module warps it back
to the original frame coordinates using an affine transform and blends it
with the background using a soft mask. This produces the final output frame.

Two implementations:
  - PutBackNumpy: Pure numpy, no state — simple but slower.
  - PutBack: Cython-accelerated with caching — used in production.
"""

import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackNumpy:
    """Pure-numpy face compositing (no caching, no Cython dependency).

    Creates a soft elliptical mask that feathers the boundary between the
    rendered face and the original background frame.
    """

    def __init__(self, mask_template_path: str | None = None):
        if mask_template_path is None:
            # Generate a default soft elliptical mask at 512x512
            single_channel_mask = get_mask(512, 512, 0.9, 0.9)
            self.blend_mask = np.concatenate([single_channel_mask] * 3, axis=2)
        else:
            mask_bgr = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.blend_mask = mask_bgr.astype(np.float32) / 255.0

    def __call__(
        self,
        original_frame: np.ndarray,
        rendered_face: np.ndarray,
        crop_to_original_matrix: np.ndarray,
    ) -> np.ndarray:
        """Composite rendered face onto original frame.

        Args:
            original_frame: Full-resolution RGB frame (H, W, 3), uint8.
            rendered_face: Rendered crop region (512, 512, 3), float32 0-255.
            crop_to_original_matrix: 3x3 affine matrix mapping crop -> original coords.

        Returns:
            Composited RGB frame (H, W, 3), uint8.
        """
        height, width = original_frame.shape[:2]
        affine_2x3 = crop_to_original_matrix[:2, :]

        warped_mask = cv2.warpAffine(
            self.blend_mask, affine_2x3, dsize=(width, height), flags=cv2.INTER_LINEAR
        ).clip(0, 1)

        warped_face = cv2.warpAffine(
            rendered_face, affine_2x3, dsize=(width, height), flags=cv2.INTER_LINEAR
        )

        # Alpha blend: mask * rendered + (1 - mask) * original
        composited = warped_mask * warped_face + (1 - warped_mask) * original_frame
        return np.clip(composited, 0, 255).astype(np.uint8)


class PutBack:
    """Cython-accelerated face compositing with mask and buffer caching.

    For single-image avatars, the affine transform is constant every frame,
    so the warped mask is computed once and cached. The Cython blend_images_cy
    function performs the alpha compositing in-place for speed.

    IMPORTANT: Returns a .copy() of the result buffer to avoid race conditions
    when frames are consumed by a separate writer thread.
    """

    def __init__(self, mask_template_path: str | None = None):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, axis=2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # Store single-channel mask for Cython blend (expects 2D float32)
        self.blend_mask_template = np.ascontiguousarray(mask)[:, :, 0]
        self.result_buffer = None

        # Cache warped mask — constant per avatar for single-image sources
        self._cached_warped_mask = None
        self._cached_matrix_bytes = None

    def __call__(
        self,
        original_frame: np.ndarray,
        rendered_face: np.ndarray,
        crop_to_original_matrix: np.ndarray,
    ) -> np.ndarray:
        """Composite rendered face onto original frame using Cython blend.

        Args:
            original_frame: Full-resolution RGB frame (H, W, 3), uint8.
            rendered_face: Rendered crop region, float32 0-255.
            crop_to_original_matrix: 3x3 affine matrix mapping crop -> original coords.

        Returns:
            Composited RGB frame (H, W, 3), uint8. Always a fresh copy.
        """
        height, width = original_frame.shape[:2]
        affine_2x3 = crop_to_original_matrix[:2, :]

        # Cache the warped mask — for single-image avatars, the transform never changes
        matrix_bytes = crop_to_original_matrix.tobytes()
        if self._cached_matrix_bytes != matrix_bytes:
            self._cached_warped_mask = cv2.warpAffine(
                self.blend_mask_template, affine_2x3,
                dsize=(width, height), flags=cv2.INTER_LINEAR,
            ).clip(0, 1)
            self._cached_matrix_bytes = matrix_bytes
            self.result_buffer = np.empty((height, width, 3), dtype=np.uint8)

        warped_face = cv2.warpAffine(
            rendered_face, affine_2x3, dsize=(width, height), flags=cv2.INTER_LINEAR,
        )

        # Cython in-place blend: result = mask * warped_face + (1-mask) * original
        blend_images_cy(self._cached_warped_mask, warped_face, original_frame, self.result_buffer)

        # Return a copy to prevent race conditions with the writer thread
        return self.result_buffer.copy()
