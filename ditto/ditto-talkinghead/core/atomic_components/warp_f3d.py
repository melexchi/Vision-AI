"""Warp Feature 3D: deforms the source appearance features using motion keypoints.

Takes the 3D feature volume from the appearance extractor (f_s) and warps it
according to the source and driving keypoints to produce a motion-aware 3D
feature volume. This is the core of the face animation — it physically moves
the neural features to match the target expression and pose.

The warp network uses grid sampling (including 5D GridSample for the 3D volume)
which requires a custom TensorRT plugin for GPU acceleration.
"""

from ..models.warp_network import WarpNetwork


class WarpF3D:
    """Thin wrapper around the WarpNetwork model.

    Provides source feature caching for single-image avatars, where the source
    appearance features (f_s) and source keypoints are constant across all frames.
    """

    def __init__(self, warp_network_cfg: dict):
        self.warp_net = WarpNetwork(**warp_network_cfg)

    def cache_source(self, feature_3d, kp_source=None):
        """Cache constant source tensors on GPU for the warp network.

        For single-image avatars, feature_3d and kp_source never change, so
        caching them avoids redundant host-to-device copies every frame.

        Args:
            feature_3d: Source appearance features (1, 32, 16, 64, 64).
            kp_source: Source keypoints (1, 21, 3), or None.
        """
        self.warp_net.cache_source(feature_3d, kp_source)

    def clear_cache(self):
        """Release cached GPU tensors."""
        self.warp_net.clear_cache()

    def __call__(self, source_features, source_keypoints, driving_keypoints):
        """Warp source features according to source->driving keypoint displacement.

        Args:
            source_features: 3D appearance features (1, 32, 16, 64, 64).
            source_keypoints: Source face keypoints (1, 21, 3).
            driving_keypoints: Target face keypoints (1, 21, 3).

        Returns:
            Warped 3D feature volume, same shape as source_features.
        """
        return self.warp_net(source_features, source_keypoints, driving_keypoints)
