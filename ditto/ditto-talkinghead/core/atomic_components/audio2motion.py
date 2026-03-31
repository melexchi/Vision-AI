"""Audio-to-Motion: converts audio conditioning into facial motion keypoint sequences.

Uses the LMDM (Latent Motion Diffusion Model) to predict per-frame motion parameters
from audio features. The motion is represented as a 265-dim vector per frame containing:
  - scale (1), pitch/yaw/roll (66 each as bin-softmax), translation (3),
    expression (63), keypoints (63).

Adjacent LMDM windows are cross-faded to avoid boundary artifacts.
"""

import numpy as np
from ..models.lmdm import LMDM

# Mapping between motion parameter dictionaries and flat numpy arrays.
# Each entry: [param_name, dict_shape, flat_size]
# Total flat dim = 1 + 66 + 66 + 66 + 3 + 63 + 63 = 328
# (but only 265 when 'kp' is excluded via ignore_keys)
MOTION_PARAM_LAYOUT = [
    ["scale", (1, 1),  1],
    ["pitch", (1, 66), 66],
    ["yaw",   (1, 66), 66],
    ["roll",  (1, 66), 66],
    ["t",     (1, 3),  3],
    ["exp",   (1, 63), 63],
    ["kp",    (1, 63), 63],
]

# Number of expression dimensions used for temporal smoothing.
# The first 202 dims cover scale+pitch+yaw+roll+t+exp (excluding kp),
# which are the ones that benefit from smoothing.
SMOOTHABLE_DIMS = 202


def motion_dict_to_array(motion_dict: dict, ignore_keys: tuple = ()) -> np.ndarray:
    """Flatten a motion parameter dict into a 1-D numpy array.

    Scale values are centered around 0 (i.e., stored as scale - 1) for
    compatibility with the LMDM's zero-centered prediction space.
    """
    parts = []
    for param_name, _, flat_size in MOTION_PARAM_LAYOUT:
        if param_name not in motion_dict or param_name in ignore_keys:
            continue
        value = motion_dict[param_name].reshape(flat_size)
        if param_name == "scale":
            value = value - 1  # Center around zero
        parts.append(value)
    return np.concatenate(parts, axis=-1)


def motion_array_to_dict(motion_array: np.ndarray, ignore_keys: tuple = ()) -> dict:
    """Reconstruct a motion parameter dict from a flat numpy array.

    Reverses the centering applied by motion_dict_to_array (scale + 1).
    """
    assert motion_array.shape[0] >= 265, f"Expected >= 265 dims, got {motion_array.shape}"
    result = {}
    offset = 0
    for param_name, dict_shape, flat_size in MOTION_PARAM_LAYOUT:
        if param_name in ignore_keys:
            continue
        value = motion_array[offset:offset + flat_size].reshape(dict_shape)
        if param_name == "scale":
            value = value + 1  # Undo zero-centering
        result[param_name] = value
        offset += flat_size
        if offset >= len(motion_array):
            break
    return result


# Backward-compatible aliases for old function name used by external code
_cvt_LP_motion_info = lambda inp, mode, ignore_keys=(): (
    motion_dict_to_array(inp, ignore_keys) if mode == "dic2arr"
    else motion_array_to_dict(inp, ignore_keys) if mode == "arr2dic"
    else (_ for _ in ()).throw(ValueError(f"Unknown mode: {mode}"))
)


class Audio2Motion:
    """Converts audio conditioning features into facial motion keypoint sequences.

    Pipeline: audio_features -> ConditionHandler -> LMDM diffusion -> motion keypoints

    The LMDM processes fixed-length windows (default 80 frames). Adjacent windows
    overlap and are cross-faded to produce smooth continuous motion.
    """

    def __init__(self, lmdm_cfg: dict):
        self.lmdm = LMDM(**lmdm_cfg)

    def setup(
        self,
        source_motion_info: dict,
        overlap_v2: int = 10,
        fix_kp_cond: int = 0,
        fix_kp_cond_dim=None,
        sampling_timesteps: int = 50,
        online_mode: bool = False,
        v_min_max_for_clip=None,
        smo_k_d: int = 3,
    ):
        """Initialize the audio-to-motion converter for a new generation session.

        Args:
            source_motion_info: Motion parameters extracted from the source avatar image.
                Used as the initial keypoint conditioning for the LMDM.
            overlap_v2: Number of frames to overlap between adjacent LMDM windows.
                Larger values = smoother transitions but more computation.
            fix_kp_cond: How often to reset keypoint conditioning to source.
                0 = never reset (accumulate drift), N = reset every N clips.
            fix_kp_cond_dim: [start, end] dimension range to preserve from
                accumulated motion when resetting. None = reset all dims.
            sampling_timesteps: Number of DDIM denoising steps. Lower = faster but noisier.
            online_mode: If True, use shorter cross-fade for lower latency.
            v_min_max_for_clip: Optional (min_array, max_array) to clamp motion values.
            smo_k_d: Temporal smoothing kernel size (1 = no smoothing).
        """
        self.smoothing_kernel_size = smo_k_d
        self.overlap_frames = overlap_v2
        self.window_length = self.lmdm.seq_frames  # Total frames per LMDM window (80)
        self.valid_output_length = self.window_length - self.overlap_frames  # New frames per window (70)

        # Cross-fade configuration — online mode uses shorter fade for lower latency
        self.online_mode = online_mode
        if self.online_mode:
            self.crossfade_length = min(self.overlap_frames, self.valid_output_length)
        else:
            self.crossfade_length = self.overlap_frames

        # Linear crossfade weights: 0 at start -> 1 at end, shape (1, crossfade_length, 1)
        self.crossfade_weights = (
            np.arange(self.crossfade_length, dtype=np.float32).reshape(1, -1, 1)
            / self.crossfade_length
        )

        self.keypoint_reset_interval = fix_kp_cond
        self.keypoint_reset_dims = fix_kp_cond_dim
        self.sampling_timesteps = sampling_timesteps

        # Value clamping to prevent extreme motion values
        self.value_clamp_range = v_min_max_for_clip
        if self.value_clamp_range is not None:
            self.clamp_min = self.value_clamp_range[0][None]  # (1, dim)
            self.clamp_max = self.value_clamp_range[1][None]

        # Extract source keypoint as initial conditioning for LMDM.
        # 'kp' is excluded because LMDM predicts it — we only condition on pose+exp.
        source_keypoint = motion_dict_to_array(
            source_motion_info, ignore_keys={"kp"}
        )[None]  # (1, dim)
        self.source_keypoint_cond = source_keypoint.copy().reshape(1, -1)
        self.current_keypoint_cond = self.source_keypoint_cond.copy()

        self.lmdm.setup(sampling_timesteps)
        self.clip_index = 0

    # ── Backward-compatible attribute aliases (used by stream_pipeline_*.py) ──

    @property
    def seq_frames(self) -> int:
        return self.window_length

    @property
    def valid_clip_len(self) -> int:
        return self.valid_output_length

    @property
    def fuse_length(self) -> int:
        return self.crossfade_length

    @property
    def smo_k_d(self) -> int:
        return self.smoothing_kernel_size

    def cvt_fmt(self, motion_sequence: np.ndarray) -> list[dict]:
        """Backward-compatible alias for convert_to_motion_dicts."""
        return self.convert_to_motion_dicts(motion_sequence)

    def _smo(self, motion_sequence, start_frame, end_frame):
        """Backward-compatible alias for _temporal_smooth."""
        return self._temporal_smooth(motion_sequence, start_frame, end_frame)

    # ── Core methods ──

    def _crossfade_windows(
        self, accumulated_motion: np.ndarray, predicted_motion: np.ndarray
    ) -> np.ndarray:
        """Cross-fade overlapping region between accumulated and newly predicted motion.

        Blending diagram (offline mode):
          accumulated:  -------
          fade region:    *****
          new window:     -------
          appended:       ^^  (valid_output_length new frames)

        Online mode uses a shorter fade at the end of the overlap for lower latency.
        """
        num_accumulated = accumulated_motion.shape[1]

        # Region in accumulated motion to blend (tail end)
        acc_fade_start = num_accumulated - self.crossfade_length
        acc_fade_end = num_accumulated

        # Region in predicted motion to blend (where overlap begins)
        pred_fade_start = self.window_length - self.valid_output_length - self.crossfade_length
        pred_fade_end = self.window_length - self.valid_output_length

        # Extract the two overlapping regions
        acc_region = accumulated_motion[:, acc_fade_start:acc_fade_end]    # (1, fade_len, dim)
        pred_region = predicted_motion[:, pred_fade_start:pred_fade_end]   # (1, fade_len, dim)

        # Linear crossfade: old * (1-alpha) + new * alpha
        blended = acc_region * (1 - self.crossfade_weights) + pred_region * self.crossfade_weights

        # Write blended region back and append new frames
        accumulated_motion[:, acc_fade_start:acc_fade_end] = blended
        new_frames = predicted_motion[:, pred_fade_end:]
        accumulated_motion = np.concatenate([accumulated_motion, new_frames], axis=1)

        return accumulated_motion

    def _update_keypoint_conditioning(
        self, accumulated_motion: np.ndarray, frame_index: int
    ):
        """Update the keypoint conditioning for the next LMDM window.

        The conditioning keypoint is taken from the last valid frame of the
        accumulated sequence. Optionally resets to source periodically to
        prevent drift accumulation over long generations.
        """
        if self.keypoint_reset_interval == 0:
            # Never reset — use last predicted frame as next conditioning
            self.current_keypoint_cond = accumulated_motion[:, frame_index - 1]
        elif self.keypoint_reset_interval > 0:
            if self.clip_index % self.keypoint_reset_interval == 0:
                # Reset to source keypoint (prevents long-term drift)
                self.current_keypoint_cond = self.source_keypoint_cond.copy()
                if self.keypoint_reset_dims is not None:
                    # Preserve specific dims from accumulated motion (e.g., expression)
                    dim_start, dim_end = self.keypoint_reset_dims
                    self.current_keypoint_cond[:, dim_start:dim_end] = (
                        accumulated_motion[:, frame_index - 1, dim_start:dim_end]
                    )
            else:
                self.current_keypoint_cond = accumulated_motion[:, frame_index - 1]

    def _temporal_smooth(
        self, motion_sequence: np.ndarray, start_frame: int, end_frame: int
    ) -> np.ndarray:
        """Apply mean-filter temporal smoothing to pose/expression dimensions.

        Only smooths the first SMOOTHABLE_DIMS dimensions (scale through expression),
        leaving keypoints unsmoothed to preserve spatial precision.
        """
        if self.smoothing_kernel_size <= 1:
            return motion_sequence

        original = motion_sequence.copy()
        num_frames = motion_sequence.shape[1]
        half_kernel = self.smoothing_kernel_size // 2

        for frame_idx in range(start_frame, end_frame):
            window_start = max(0, frame_idx - half_kernel)
            window_end = min(num_frames, frame_idx + half_kernel + 1)
            motion_sequence[:, frame_idx, :SMOOTHABLE_DIMS] = np.mean(
                original[:, window_start:window_end, :SMOOTHABLE_DIMS], axis=1
            )
        return motion_sequence

    def __call__(
        self, audio_conditioning: np.ndarray, accumulated_motion: np.ndarray | None = None
    ) -> np.ndarray:
        """Run one LMDM window: predict motion from audio and merge with accumulated.

        Args:
            audio_conditioning: (1, window_length, audio_dim) conditioning features.
            accumulated_motion: Previous accumulated motion sequence, or None for first window.

        Returns:
            Updated accumulated motion sequence: (1, total_frames, 265).
        """
        predicted_motion = self.lmdm(
            self.current_keypoint_cond, audio_conditioning, self.sampling_timesteps
        )

        if accumulated_motion is None:
            # First window — use prediction directly
            accumulated_motion = predicted_motion  # (1, window_length, dim)
            accumulated_motion = self._temporal_smooth(
                accumulated_motion, 0, accumulated_motion.shape[1]
            )
        else:
            # Subsequent windows — crossfade overlap region
            accumulated_motion = self._crossfade_windows(accumulated_motion, predicted_motion)
            # Smooth only the newly added region + crossfade zone
            smooth_start = accumulated_motion.shape[1] - self.valid_output_length - self.crossfade_length
            smooth_end = accumulated_motion.shape[1] - self.valid_output_length + 1
            accumulated_motion = self._temporal_smooth(
                accumulated_motion, smooth_start, smooth_end
            )

        self.clip_index += 1

        # Update conditioning: take frame just before the overlap region
        conditioning_frame_idx = accumulated_motion.shape[1] - self.overlap_frames
        self._update_keypoint_conditioning(accumulated_motion, conditioning_frame_idx)

        return accumulated_motion

    def convert_to_motion_dicts(self, motion_sequence: np.ndarray) -> list[dict]:
        """Convert a (1, N, 265) motion array into a list of per-frame motion dicts.

        Each dict contains: {scale, pitch, yaw, roll, t, exp, kp} with proper shapes.
        Optionally clamps values to the configured range.
        """
        if self.value_clamp_range is not None:
            frames = np.clip(motion_sequence[0], self.clamp_min, self.clamp_max)
        else:
            frames = motion_sequence[0]

        motion_dicts = []
        for frame_idx in range(frames.shape[0]):
            motion_dict = motion_array_to_dict(frames[frame_idx])
            motion_dicts.append(motion_dict)
        return motion_dicts
