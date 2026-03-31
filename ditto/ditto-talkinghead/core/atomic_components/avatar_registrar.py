"""Avatar Registrar: processes source images/videos into animation-ready feature sets.

This is the entry point for preparing a new avatar. Given a source image or video,
it runs the full feature extraction pipeline and produces a "source_info" dict
containing everything needed to animate the face:
  - Per-frame motion parameters (pose, expression, scale, keypoints)
  - Per-frame 3D appearance features (for the warp network)
  - Per-frame crop transforms (for compositing back onto the original image)
  - Per-frame eye tracking data (openness, gaze direction)
  - Shape coefficients (face identity)
  - Original RGB frames (for background compositing)

For single-image avatars, all lists have length 1.
For video avatars, landmarks are tracked across frames for temporal consistency.
"""

import numpy as np

from .loader import load_source_frames
from .source2info import Source2Info


def temporal_mean_filter(array: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a 1-D mean filter along the first axis for temporal smoothing.

    Used to smooth motion parameters extracted from video sources,
    reducing jitter from per-frame estimation noise.
    """
    num_frames = array.shape[0]
    half_kernel = kernel_size // 2
    smoothed = []
    for frame_idx in range(num_frames):
        window_start = max(0, frame_idx - half_kernel)
        window_end = min(num_frames, frame_idx + half_kernel + 1)
        smoothed.append(array[window_start:window_end].mean(axis=0))
    return np.stack(smoothed, axis=0)


def smooth_motion_info_list(
    motion_info_list: list[dict],
    ignore_keys: tuple = (),
    smoothing_kernel: int = 13,
) -> list[dict]:
    """Apply temporal smoothing to a list of per-frame motion parameter dicts.

    Smooths all numeric parameters across frames using a mean filter,
    except for keys listed in ignore_keys.
    """
    param_names = motion_info_list[0].keys()
    num_frames = len(motion_info_list)

    # Stack each parameter across frames, smooth, then unstack
    smoothed_params = {}
    for param_name in param_names:
        values = [motion_info_list[idx][param_name] for idx in range(num_frames)]
        if param_name not in ignore_keys:
            values = np.stack(values, axis=0)
            smoothed_values = temporal_mean_filter(values, smoothing_kernel)
        else:
            smoothed_values = values
        smoothed_params[param_name] = smoothed_values

    # Reconstruct per-frame dicts
    return [
        {param_name: smoothed_params[param_name][idx] for param_name in param_names}
        for idx in range(num_frames)
    ]


# Backward-compatible alias
smooth_x_s_info_lst = smooth_motion_info_list


class AvatarRegistrar:
    """Registers a source image/video for animation.

    Runs face detection, landmark extraction, appearance feature extraction,
    and motion parameter estimation on each frame of the source media.

    For single images: produces a 1-frame source_info dict.
    For videos: tracks landmarks across frames and optionally smooths parameters.
    """

    def __init__(
        self,
        insightface_det_cfg: dict,
        landmark106_cfg: dict,
        landmark203_cfg: dict,
        landmark478_cfg: dict,
        appearance_extractor_cfg: dict,
        motion_extractor_cfg: dict,
    ):
        self.source_to_info = Source2Info(
            insightface_det_cfg,
            landmark106_cfg,
            landmark203_cfg,
            landmark478_cfg,
            appearance_extractor_cfg,
            motion_extractor_cfg,
        )

    def register(
        self,
        source_path: str,
        max_dim: int = 1920,
        n_frames: int = -1,
        **crop_kwargs,
    ) -> dict:
        """Register an avatar from an image or video file.

        Args:
            source_path: Path to source image or video.
            max_dim: Maximum resolution for the source (default 1920px).
            n_frames: Maximum video frames to process (-1 = all).
            **crop_kwargs: Crop configuration:
                crop_scale: Face crop scale (default 2.3).
                crop_vx_ratio: Horizontal crop offset (default 0).
                crop_vy_ratio: Vertical crop offset (default -0.125).
                crop_flag_do_rot: Whether to rotate for alignment (default True).

        Returns:
            source_info dict containing:
                x_s_info_lst: List of per-frame motion parameter dicts.
                f_s_lst: List of per-frame 3D appearance feature arrays.
                M_c2o_lst: List of per-frame crop-to-original affine matrices.
                eye_open_lst: List of per-frame eye openness arrays (1, 2).
                eye_ball_lst: List of per-frame eye gaze arrays (1, 6).
                sc: Shape coefficients (63-dim canonical keypoints from frame 0).
                is_image_flag: True if source was a single image.
                img_rgb_lst: List of original RGB frames.
        """
        frame_list, is_single_image = load_source_frames(
            source_path, max_dim=max_dim, n_frames=n_frames,
        )

        source_info = {
            "x_s_info_lst": [],
            "f_s_lst": [],
            "M_c2o_lst": [],
            "eye_open_lst": [],
            "eye_ball_lst": [],
        }

        output_keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        previous_landmarks = None

        for frame_rgb in frame_list:
            frame_info = self.source_to_info(frame_rgb, previous_landmarks, **crop_kwargs)
            for key in output_keys:
                source_info[f"{key}_lst"].append(frame_info[key])
            previous_landmarks = frame_info["lmk203"]

        # Shape coefficients: canonical keypoints from the first frame
        shape_coefficients = source_info["x_s_info_lst"][0]["kp"].flatten()

        source_info["sc"] = shape_coefficients
        source_info["is_image_flag"] = is_single_image
        source_info["img_rgb_lst"] = frame_list

        return source_info

    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)
