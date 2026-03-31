"""Motion Stitch: combines source and driving motion, applies stitching correction.

This is where source pose + driving expression are merged into the final keypoints.
The process:
  1. Mix source and driving motion parameters (relative or absolute mode)
  2. Fix expression: only lip + optional eye movements come from driving; rest from source
  3. Apply gaze correction to keep eyes tracking naturally
  4. Transform keypoints using rotation matrix + translation
  5. Run the stitching network to fix boundary artifacts between face and background

The stitching network is a small MLP that adjusts keypoints to prevent
visible seams where the animated face meets the static background.
"""

import copy
import random
import numpy as np
from scipy.special import softmax

from ..models.stitch_network import StitchNetwork

# Expression landmark group indices (out of 21 3D landmarks)
EYE_LANDMARK_INDICES = [11, 13, 15, 16, 18]
LIP_LANDMARK_INDICES = [6, 12, 14, 17, 19, 20]


def apply_motion_controls(driving_motion: dict, **control_kwargs) -> dict:
    """Apply user-specified motion control overrides.

    Supports:
      - delta_pitch/yaw/roll: Add offset to head rotation (in degrees)
      - alpha_pitch/yaw/roll: Scale head rotation
      - delta_exp: Add offset to expression coefficients
    """
    # Additive pose offsets
    for control_key in ["delta_pitch", "delta_yaw", "delta_roll"]:
        if control_key in control_kwargs:
            param_name = control_key[6:]  # "delta_pitch" -> "pitch"
            driving_motion[param_name] = (
                bin66_to_degree(driving_motion[param_name]) + control_kwargs[control_key]
            )

    # Multiplicative pose scaling
    for control_key in ["alpha_pitch", "alpha_yaw", "alpha_roll"]:
        if control_key in control_kwargs:
            param_name = control_key[6:]
            driving_motion[param_name] = driving_motion[param_name] * control_kwargs[control_key]

    # Additive expression offset
    if "delta_exp" in control_kwargs:
        driving_motion["exp"] = driving_motion["exp"] + control_kwargs["delta_exp"]

    return driving_motion


def fade_motion(motion: dict, target: dict, alpha: float, keys: list | None = None) -> dict:
    """Linearly interpolate motion toward a target state.

    Used for fade-in/fade-out effects at segment boundaries.
    alpha=1 means full motion, alpha=0 means full target.
    """
    if keys is None:
        keys = motion.keys()
    for param_name in keys:
        if param_name == "kp":
            continue  # Keypoints are derived, not directly faded
        motion[param_name] = motion[param_name] * alpha + target[param_name] * (1 - alpha)
    return motion


def apply_vad_blending(driving_motion: dict, source_motion: dict, blend_alpha: float) -> dict:
    """Blend expression toward source during silence (VAD-based).

    When the speaker is silent (low VAD), blend expression back toward
    the neutral source expression to reduce phantom lip movement.
    """
    driving_motion["exp"] = (
        driving_motion["exp"] * blend_alpha
        + source_motion["exp"] * (1 - blend_alpha)
    )
    return driving_motion


def merge_source_and_driving(
    source_info: dict,
    driving_info: dict,
    active_driving_keys: tuple | dict = ("exp", "pitch", "yaw", "roll", "t"),
    reference_driving: dict | None = None,
) -> dict:
    """Merge source and driving motion parameters.

    If reference_driving (d0) is provided, uses relative motion:
        result[k] = source[k] + (driving[k] - d0[k]) * weight
    Otherwise uses driving values directly for active keys.

    Args:
        source_info: Source avatar's motion parameters.
        driving_info: LMDM-predicted driving motion parameters.
        active_driving_keys: Which parameters to take from driving.
            Can be a dict {name: weight} for weighted blending.
        reference_driving: First-frame driving motion for relative mode.
    """
    if reference_driving is not None:
        if isinstance(active_driving_keys, dict):
            driving_info = {
                key: source_info[key] + (value - reference_driving[key]) * active_driving_keys.get(key, 1)
                for key, value in driving_info.items()
            }
        else:
            driving_info = {
                key: source_info[key] + (value - reference_driving[key])
                for key, value in driving_info.items()
            }

    # Fill in non-driven parameters from source
    for key, value in source_info.items():
        if key not in driving_info or key not in active_driving_keys:
            driving_info[key] = value

    # Apply per-key weights when d0 is not used
    if isinstance(active_driving_keys, dict) and reference_driving is None:
        for key, weight in active_driving_keys.items():
            driving_info[key] *= weight

    return driving_info


def generate_blink_schedule(
    total_frames: int,
    blink_duration: int = 15,
    open_intervals: int | list = -1,
) -> list[int]:
    """Generate a blink animation schedule.

    Returns a list of length total_frames where each value is the blink
    animation frame index (0 = fully open, >0 = closing/opening).

    Args:
        total_frames: Total number of animation frames.
        blink_duration: Number of frames per blink animation.
        open_intervals: Frames between blinks.
            -1: No blinking.
            0: Random intervals (60-100 frames).
            >0: Fixed interval.
            list: Cycle through specified intervals.
    """
    MIN_OPEN_FRAMES = 60
    MAX_OPEN_FRAMES = 100

    schedule = [0] * total_frames

    if isinstance(open_intervals, int):
        if open_intervals < 0:
            return schedule  # No blinking
        elif open_intervals > 0:
            interval_list = [open_intervals]
        else:
            interval_list = []  # Random intervals
    elif isinstance(open_intervals, list):
        interval_list = open_intervals
    else:
        raise ValueError(f"Invalid open_intervals type: {type(open_intervals)}")

    blink_frames = list(range(blink_duration))
    first_interval = interval_list[0] if interval_list else random.randint(MIN_OPEN_FRAMES, MAX_OPEN_FRAMES)
    last_interval = interval_list[-1] if interval_list else random.randint(MIN_OPEN_FRAMES, MAX_OPEN_FRAMES)

    max_blink_start = total_frames - max(last_interval, blink_duration)
    current_frame = first_interval
    interval_index = 1

    while current_frame < max_blink_start:
        schedule[current_frame:current_frame + blink_duration] = blink_frames

        if interval_list:
            next_interval = interval_list[interval_index % len(interval_list)]
            interval_index += 1
        else:
            next_interval = random.randint(MIN_OPEN_FRAMES, MAX_OPEN_FRAMES)

        current_frame += blink_duration + next_interval

    return schedule


def apply_expression_mask(
    driving_info: dict,
    source_info: dict,
    lip_weight: np.ndarray,
    source_weight: np.ndarray,
    eye_delta_weight: np.ndarray,
    eye_delta: float | np.ndarray = 0,
) -> dict:
    """Apply weighted expression mixing between driving and source.

    The expression is a 63-dim vector (21 landmarks * 3 coordinates).
    Different landmark groups get different weights:
      - Lip landmarks: fully from driving (lip_weight)
      - Eye landmarks: from eye_delta or source (eye_delta_weight)
      - Other landmarks: from source (source_weight)

    This ensures only speech-related motion comes from the audio-driven model,
    while the rest of the face maintains the source appearance.
    """
    driving_info["exp"] = (
        driving_info["exp"] * lip_weight
        + source_info["exp"] * source_weight
        + eye_delta * eye_delta_weight
    )
    return driving_info


def bin66_to_degree(prediction: np.ndarray) -> np.ndarray:
    """Convert 66-bin softmax head pose prediction to degrees.

    The model predicts head rotation as a probability distribution over 66 bins,
    each representing a 3-degree range. The expected value gives the rotation angle.
    Range: [-97.5, 97.5] degrees.
    """
    if prediction.ndim > 1 and prediction.shape[1] == 66:
        bin_indices = np.arange(66).astype(np.float32)
        probabilities = softmax(prediction, axis=1)
        degree = np.sum(probabilities * bin_indices, axis=1) * 3 - 97.5
        return degree
    return prediction


def compute_eye_gaze_offset(expression: np.ndarray, delta_x: float = 0, delta_y: float = 0) -> np.ndarray:
    """Apply gaze direction offset to eye expression coefficients.

    Modifies specific expression dimensions that control eye ball position.
    Asymmetric x-scaling simulates natural eye convergence.
    """
    if delta_x > 0:
        expression[0, 33] += delta_x * 0.0007
        expression[0, 45] += delta_x * 0.001
    else:
        expression[0, 33] += delta_x * 0.001
        expression[0, 45] += delta_x * 0.0007

    expression[0, 34] += delta_y * -0.001
    expression[0, 46] += delta_y * -0.001
    return expression


def correct_gaze_direction(source_pose: list, driving_info: dict) -> dict:
    """Correct eye gaze to compensate for head rotation changes.

    When the head turns, the eyes should counter-rotate to maintain
    the appearance of looking at the camera. This uses the delta between
    driving and source head pose to compute a compensating gaze offset.
    """
    gaze_x_ratio = 0.26
    gaze_y_ratio = 0.28

    source_yaw, source_pitch = source_pose
    driving_yaw = bin66_to_degree(driving_info["yaw"]).item()
    driving_pitch = bin66_to_degree(driving_info["pitch"]).item()

    delta_yaw = driving_yaw - source_yaw
    delta_pitch = driving_pitch - source_pitch

    gaze_offset_x = delta_yaw * gaze_x_ratio
    gaze_offset_y = delta_pitch * gaze_y_ratio

    driving_info["exp"] = compute_eye_gaze_offset(driving_info["exp"], gaze_offset_x, gaze_offset_y)
    return driving_info


def compute_rotation_matrix(
    pitch_deg: np.ndarray, yaw_deg: np.ndarray, roll_deg: np.ndarray
) -> np.ndarray:
    """Compute 3x3 rotation matrix from Euler angles in degrees.

    Uses the ZYX convention (roll -> yaw -> pitch) and returns the transpose
    for the keypoint transformation equation.
    """
    pitch = pitch_deg / 180 * np.pi
    yaw = yaw_deg / 180 * np.pi
    roll = roll_deg / 180 * np.pi

    if pitch.ndim == 1:
        pitch = pitch[:, None]
    if yaw.ndim == 1:
        yaw = yaw[:, None]
    if roll.ndim == 1:
        roll = roll[:, None]

    batch_size = pitch.shape[0]
    ones = np.ones((batch_size, 1), dtype=np.float32)
    zeros = np.zeros((batch_size, 1), dtype=np.float32)

    # Rotation around X axis (pitch)
    rot_x = np.concatenate([
        ones, zeros, zeros,
        zeros, np.cos(pitch), -np.sin(pitch),
        zeros, np.sin(pitch), np.cos(pitch),
    ], axis=1).reshape(batch_size, 3, 3)

    # Rotation around Y axis (yaw)
    rot_y = np.concatenate([
        np.cos(yaw), zeros, np.sin(yaw),
        zeros, ones, zeros,
        -np.sin(yaw), zeros, np.cos(yaw),
    ], axis=1).reshape(batch_size, 3, 3)

    # Rotation around Z axis (roll)
    rot_z = np.concatenate([
        np.cos(roll), -np.sin(roll), zeros,
        np.sin(roll), np.cos(roll), zeros,
        zeros, zeros, ones,
    ], axis=1).reshape(batch_size, 3, 3)

    rotation = np.matmul(np.matmul(rot_z, rot_y), rot_x)
    return np.transpose(rotation, (0, 2, 1))


def transform_keypoint(keypoint_info: dict) -> np.ndarray:
    """Transform implicit keypoints with pose, scale, expression, and translation.

    Implements the equation: s * (R * kp + exp) + t
    where s=scale, R=rotation matrix, kp=canonical keypoints, exp=expression, t=translation.

    Args:
        keypoint_info: Dict with keys: kp, pitch, yaw, roll, t, exp, scale.

    Returns:
        Transformed keypoints, shape (batch, num_keypoints, 3).
    """
    keypoints = keypoint_info["kp"]  # (batch, num_kp, 3) or (batch, num_kp*3)
    pitch = bin66_to_degree(keypoint_info["pitch"])
    yaw = bin66_to_degree(keypoint_info["yaw"])
    roll = bin66_to_degree(keypoint_info["roll"])
    translation = keypoint_info["t"]
    expression = keypoint_info["exp"]
    scale = keypoint_info["scale"]

    batch_size = keypoints.shape[0]
    if keypoints.ndim == 2:
        num_keypoints = keypoints.shape[1] // 3
    else:
        num_keypoints = keypoints.shape[1]

    rotation_matrix = compute_rotation_matrix(pitch, yaw, roll)  # (batch, 3, 3)

    # s * (R * kp + exp) + t
    transformed = np.matmul(keypoints.reshape(batch_size, num_keypoints, 3), rotation_matrix)
    transformed += expression.reshape(batch_size, num_keypoints, 3)
    transformed *= scale[..., None]  # (batch, kp, 3) * (batch, 1, 1)
    transformed[:, :, 0:2] += translation[:, None, 0:2]  # Only apply tx, ty (ignore tz)

    return transformed


class MotionStitch:
    """Merges source and driving motion, applies expression masks, and runs stitching.

    This is the central motion processing stage:
      1. Merge source pose with driving expression (relative to d0 reference)
      2. Mask expression: only lip (+optional eye) from driving, rest from source
      3. Apply blink animation overlay
      4. Apply user controls (pose offsets, VAD blending, fade effects)
      5. Correct gaze direction for head rotation
      6. Transform keypoints with rotation + translation
      7. Run stitching network to fix face-background boundary
    """

    def __init__(self, stitch_network_cfg: dict):
        self.stitch_net = StitchNetwork(**stitch_network_cfg)

    def set_Nd(self, num_driving_frames: int = -1):
        """Update the total frame count (for blink schedule regeneration)."""
        if num_driving_frames == self.num_driving_frames:
            return

        self.num_driving_frames = num_driving_frames
        if self.drive_eye and self.blink_animation_frames is not None:
            total = 3000 if self.num_driving_frames == -1 else self.num_driving_frames
            self.blink_schedule = generate_blink_schedule(
                total, len(self.blink_animation_frames), self.blink_open_interval
            )

    # Backward-compatible property
    @property
    def N_d(self):
        return self.num_driving_frames

    @N_d.setter
    def N_d(self, value):
        self.num_driving_frames = value

    def setup(
        self,
        N_d: int = -1,
        use_d_keys=None,
        relative_d: bool = True,
        drive_eye=None,
        delta_eye_arr=None,
        delta_eye_open_n=-1,
        fade_out_keys: tuple = ("exp",),
        fade_type: str = "",
        flag_stitching: bool = True,
        is_image_flag: bool = True,
        x_s_info=None,
        d0=None,
        ch_info=None,
        overall_ctrl_info=None,
    ):
        """Configure the motion stitching pipeline.

        Args:
            N_d: Total driving frames (-1 = unknown/streaming).
            use_d_keys: Which motion params to take from driving.
                Tuple of names, or dict {name: weight}.
            relative_d: Use relative motion (driving - d0 + source).
            drive_eye: Whether to animate eyes from driving signal.
            delta_eye_arr: Pre-computed blink animation frames.
            delta_eye_open_n: Blink interval (-1=none, 0=random, >0=fixed, list=cycle).
            fade_out_keys: Which params to fade at segment boundaries.
            fade_type: Fade target — "" (none), "d0" (first frame), "s" (source).
            flag_stitching: Whether to run the stitching network.
            is_image_flag: True for single-image source, False for video.
            x_s_info: Source motion parameters.
            d0: Reference driving frame for relative motion.
            ch_info: Cross-identity conditioning info.
            overall_ctrl_info: Default motion control overrides.
        """
        self.is_image_flag = is_image_flag

        # Which driving parameters to use
        if use_d_keys is None:
            if self.is_image_flag:
                self.active_driving_keys = ("exp", "pitch", "yaw", "roll", "t")
            else:
                self.active_driving_keys = ("exp",)
        else:
            self.active_driving_keys = use_d_keys

        # Eye animation mode
        if drive_eye is None:
            self.drive_eye = self.is_image_flag
        else:
            self.drive_eye = drive_eye

        self.num_driving_frames = N_d
        self.use_relative_motion = relative_d
        self.blink_animation_frames = delta_eye_arr
        self.blink_open_interval = delta_eye_open_n
        self.fade_out_keys = fade_out_keys
        self.fade_type = fade_type
        self.enable_stitching = flag_stitching

        # Build expression mixing masks
        # lip_mask: 1 for lip landmarks, 0 elsewhere
        # eye_delta_mask: 1 for eye landmarks (when using blink animation)
        # source_mask: 1 for everything else
        lip_mask = np.zeros((21, 3), dtype=np.float32)
        lip_mask[LIP_LANDMARK_INDICES] = 1
        eye_delta_mask = 0

        if self.drive_eye:
            if self.blink_animation_frames is None:
                # Take eye motion from driving signal
                lip_mask[EYE_LANDMARK_INDICES] = 1
            else:
                # Use pre-computed blink animation
                eye_delta_mask = np.zeros((21, 3), dtype=np.float32)
                eye_delta_mask[EYE_LANDMARK_INDICES] = 1
                eye_delta_mask = eye_delta_mask.reshape(1, -1)

        lip_mask = lip_mask.reshape(1, -1)

        # Final masks: lip_weight * (1 - eye_delta) for driving expression,
        # rest goes to source, eye_delta goes to blink animation
        self.expression_lip_weight = lip_mask * (1 - eye_delta_mask)
        self.expression_source_weight = (1 - lip_mask) + lip_mask * eye_delta_mask
        self.expression_eye_weight = eye_delta_mask

        # Generate blink schedule
        if self.drive_eye and self.blink_animation_frames is not None:
            total = 3000 if self.num_driving_frames == -1 else self.num_driving_frames
            self.blink_schedule = generate_blink_schedule(
                total, len(self.blink_animation_frames), self.blink_open_interval
            )

        # Cache source pose and transformed keypoints (constant for image sources)
        self.source_pose = None
        self.source_keypoints_transformed = None
        self.fade_target = None

        if self.is_image_flag and x_s_info is not None:
            source_yaw = bin66_to_degree(x_s_info["yaw"]).item()
            source_pitch = bin66_to_degree(x_s_info["pitch"]).item()
            self.source_pose = [source_yaw, source_pitch]
            self.source_keypoints_transformed = transform_keypoint(x_s_info)

            if self.fade_type == "s":
                self.fade_target = copy.deepcopy(x_s_info)

        # Cross-identity scale correction
        if ch_info is not None:
            self.cross_id_scale = ch_info["x_s_info_lst"][0]["scale"].item()
            if x_s_info is not None:
                source_scale = x_s_info["scale"].item()
                self.scale_ratio = self.cross_id_scale / source_scale
                self._apply_scale_ratio(self.scale_ratio)
            else:
                self.scale_ratio = None
        else:
            self.scale_ratio = 1

        self.overall_ctrl_info = overall_ctrl_info
        self.d0 = d0
        self.frame_index = 0

    def _apply_scale_ratio(self, scale_ratio: float = 1):
        """Adjust driving key weights for cross-identity scale differences."""
        if scale_ratio == 1:
            return
        scaleable_keys = {"exp", "pitch", "yaw", "roll"}
        if isinstance(self.active_driving_keys, dict):
            self.active_driving_keys = {
                key: weight * (scale_ratio if key in scaleable_keys else 1)
                for key, weight in self.active_driving_keys.items()
            }
        else:
            self.active_driving_keys = {
                key: scale_ratio if key in scaleable_keys else 1
                for key in self.active_driving_keys
            }

    @staticmethod
    def _merge_kwargs(default_kwargs, runtime_kwargs):
        """Merge default control kwargs with per-call overrides."""
        if default_kwargs is None:
            return runtime_kwargs
        for key, value in default_kwargs.items():
            if key not in runtime_kwargs:
                runtime_kwargs[key] = value
        return runtime_kwargs

    def __call__(self, source_info: dict, driving_info: dict, **control_kwargs) -> tuple:
        """Process one frame: merge motion, apply controls, transform keypoints.

        Args:
            source_info: Source avatar's motion parameters for this frame.
            driving_info: LMDM-predicted driving motion for this frame.
            **control_kwargs: Per-frame overrides (fade_alpha, delta_pitch, etc.)

        Returns:
            (source_keypoints, driving_keypoints): Both shape (1, 21, 3),
            ready for the warp network.
        """
        control_kwargs = self._merge_kwargs(self.overall_ctrl_info, control_kwargs)

        # Lazy cross-identity scale initialization
        if self.scale_ratio is None:
            source_scale = source_info["scale"].item()
            self.scale_ratio = self.cross_id_scale / source_scale
            self._apply_scale_ratio(self.scale_ratio)

        # Set d0 reference from first driving frame (for relative motion)
        if self.use_relative_motion and self.d0 is None:
            self.d0 = copy.deepcopy(driving_info)

        # Step 1: Merge source and driving parameters
        driving_info = merge_source_and_driving(
            source_info, driving_info, self.active_driving_keys, self.d0,
        )

        # Step 2: Apply expression mask (lip from driving, rest from source)
        eye_delta = 0
        if self.drive_eye and self.blink_animation_frames is not None:
            blink_frame_idx = self.blink_schedule[self.frame_index % len(self.blink_schedule)]
            eye_delta = self.blink_animation_frames[blink_frame_idx][None]

        driving_info = apply_expression_mask(
            driving_info, source_info,
            self.expression_lip_weight,
            self.expression_source_weight,
            self.expression_eye_weight,
            eye_delta,
        )

        # Step 3: VAD-based expression blending (reduce lip motion during silence)
        if control_kwargs.get("vad_alpha", 1) < 1:
            driving_info = apply_vad_blending(
                driving_info, source_info, control_kwargs.get("vad_alpha", 1)
            )

        # Step 4: Apply user motion controls
        driving_info = apply_motion_controls(driving_info, **control_kwargs)

        # Step 5: Set fade target from first driving frame
        if self.fade_type == "d0" and self.fade_target is None:
            self.fade_target = copy.deepcopy(driving_info)

        # Step 6: Apply fade effect at segment boundaries
        if "fade_alpha" in control_kwargs and self.fade_type in ["d0", "s"]:
            fade_alpha = control_kwargs["fade_alpha"]
            fade_keys = control_kwargs.get("fade_out_keys", self.fade_out_keys)
            if self.fade_type == "d0":
                target = self.fade_target
            elif self.fade_type == "s":
                if self.fade_target is not None:
                    target = self.fade_target
                else:
                    target = copy.deepcopy(source_info)
                    if self.is_image_flag:
                        self.fade_target = target
            driving_info = fade_motion(driving_info, target, fade_alpha, fade_keys)

        # Step 7: Correct gaze for head rotation
        if self.drive_eye:
            if self.source_pose is None:
                source_yaw = bin66_to_degree(source_info["yaw"]).item()
                source_pitch = bin66_to_degree(source_info["pitch"]).item()
                self.source_pose = [source_yaw, source_pitch]
            driving_info = correct_gaze_direction(self.source_pose, driving_info)

        # Step 8: Transform keypoints (apply rotation + translation)
        if self.source_keypoints_transformed is not None:
            source_keypoints = self.source_keypoints_transformed
        else:
            source_keypoints = transform_keypoint(source_info)
            if self.is_image_flag:
                self.source_keypoints_transformed = source_keypoints

        driving_keypoints = transform_keypoint(driving_info)

        # Step 9: Run stitching network to fix face-background boundary
        if self.enable_stitching:
            driving_keypoints = self.stitch_net(source_keypoints, driving_keypoints)

        self.frame_index += 1
        return source_keypoints, driving_keypoints
