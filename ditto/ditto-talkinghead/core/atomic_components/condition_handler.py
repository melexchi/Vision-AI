"""Condition Handler: builds the full conditioning vector for the LMDM diffusion model.

The LMDM needs more than just audio features — it also uses:
  - Emotion labels (8-class softmax: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, Contempt)
  - Eye openness (left/right, 2 values per frame)
  - Eye ball position (left/right x 3 axes, 6 values per frame)
  - Shape coefficients (63-dim face identity keypoints, constant per avatar)

This module assembles all conditioning signals into a single feature vector per frame
that the LMDM consumes alongside the noised motion input.
"""

import copy
import numpy as np
from scipy.special import softmax


# Emotion class names for reference (index -> label)
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise", "Contempt",
]


def get_emotion_distribution(emotion_index: int | list[int] = 6) -> np.ndarray:
    """Create a softmax emotion distribution from one or more emotion indices.

    Uses a temperature-scaled one-hot encoding: active emotions get weight 8,
    then softmax normalizes. This gives a peaked but smooth distribution.

    Args:
        emotion_index: Single emotion index (0-7) or list of active emotions.

    Returns:
        Probability distribution over 8 emotion classes, shape (8,).
    """
    logits = np.zeros(8, dtype=np.float32)
    if isinstance(emotion_index, (list, tuple)):
        for idx in emotion_index:
            logits[idx] = 8
    else:
        logits[emotion_index] = 8
    return softmax(logits)


def mirror_index(index: int, sequence_length: int) -> int:
    """Map an unbounded index into a sequence using ping-pong (mirror) looping.

    For a sequence of length 5: 0,1,2,3,4,3,2,1,0,1,2,3,4,...
    This avoids discontinuities at loop boundaries in video template frames.
    """
    period = index // sequence_length
    position = index % sequence_length
    if period % 2 == 0:
        return position  # Forward pass
    else:
        return sequence_length - position - 1  # Backward pass


# Backward-compatible alias for external imports
_mirror_index = mirror_index


class ConditionHandler:
    """Assembles per-frame conditioning vectors for the LMDM diffusion model.

    Concatenates audio features with emotion, eye state, and shape coefficient
    signals to form the complete conditioning input:
        [audio_feat(1024) | emotion(8) | eye_open(2) | eye_ball(6) | shape_coeff(63)]
        = 1103-dim conditioning vector per frame (when all signals enabled).

    For single-image avatars, emotion/eye/shape are typically constant.
    For video-driven generation, they can vary per frame.
    """

    def __init__(
        self,
        use_emo: bool = True,
        use_sc: bool = True,
        use_eye_open: bool = True,
        use_eye_ball: bool = True,
        seq_frames: int = 80,
    ):
        self.use_emotion = use_emo
        self.use_shape_coeff = use_sc
        self.use_eye_open = use_eye_open
        self.use_eye_ball = use_eye_ball
        self.window_length = seq_frames

    def setup(self, source_info: dict, emotion, eye_f0_mode: bool = False, ch_info=None):
        """Initialize conditioning from source avatar info.

        Args:
            source_info: Registration output containing x_s_info_lst, eye_open_lst, etc.
            emotion: Emotion specification — int (single label), list[int] (multi-label),
                list[list[int]] (per-frame labels), or (N, 8) numpy array.
            eye_f0_mode: If True, freeze eye state to first frame (for video sources).
            ch_info: Optional override for conditioning source (cross-identity transfer).
        """
        if ch_info is None:
            conditioning_source = copy.deepcopy(source_info)
        else:
            conditioning_source = ch_info

        self.eye_f0_mode = eye_f0_mode
        self.x_s_info_0 = conditioning_source["x_s_info_lst"][0]

        # Shape coefficients (face identity) — constant per avatar
        if self.use_shape_coeff:
            self.shape_coefficients = conditioning_source["sc"]  # (63,)
            self.shape_coeff_window = np.stack(
                [self.shape_coefficients] * self.window_length, axis=0
            )

        # Eye openness — (N, 2) array of left/right eye openness values
        if self.use_eye_open:
            self.eye_open_sequence = np.concatenate(conditioning_source["eye_open_lst"], axis=0)
            self.num_eye_open_frames = len(self.eye_open_sequence)
            if self.num_eye_open_frames == 1 or self.eye_f0_mode:
                # Single-image or frozen mode: replicate first frame
                self.eye_open_window = np.stack(
                    [self.eye_open_sequence[0]] * self.window_length, axis=0
                )
            else:
                self.eye_open_window = None  # Will be computed per-call

        # Eye ball position — (N, 6) array of gaze direction
        if self.use_eye_ball:
            self.eye_ball_sequence = np.concatenate(conditioning_source["eye_ball_lst"], axis=0)
            self.num_eye_ball_frames = len(self.eye_ball_sequence)
            if self.num_eye_ball_frames == 1 or self.eye_f0_mode:
                self.eye_ball_window = np.stack(
                    [self.eye_ball_sequence[0]] * self.window_length, axis=0
                )
            else:
                self.eye_ball_window = None

        # Emotion distribution — (N, 8) softmax probabilities
        if self.use_emotion:
            self.emotion_sequence = self._parse_emotion(emotion)
            self.num_emotion_frames = len(self.emotion_sequence)
            if self.num_emotion_frames == 1:
                self.emotion_window = np.concatenate(
                    [self.emotion_sequence] * self.window_length, axis=0
                )
            else:
                self.emotion_window = None

    @staticmethod
    def _parse_emotion(emotion, target_length: int = -1) -> np.ndarray:
        """Parse various emotion input formats into a (N, 8) array.

        Supported formats:
            - int: Single emotion label (e.g., 4 for "Neutral")
            - list[int]: Multi-label (e.g., [3, 4] for "Happy+Neutral")
            - list[list[int]]: Per-frame labels
            - np.ndarray (N, 8): Pre-computed distributions
        """
        if isinstance(emotion, np.ndarray) and emotion.ndim == 2 and emotion.shape[1] == 8:
            emotion_array = emotion
        elif isinstance(emotion, int) and 0 <= emotion < 8:
            emotion_array = get_emotion_distribution(emotion).reshape(1, 8)
        elif isinstance(emotion, (list, tuple)) and 0 < len(emotion) < 8 and isinstance(emotion[0], int):
            emotion_array = get_emotion_distribution(emotion).reshape(1, 8)
        elif isinstance(emotion, list) and emotion and isinstance(emotion[0], (list, tuple)):
            emotion_array = np.stack([get_emotion_distribution(frame_emo) for frame_emo in emotion], axis=0)
        else:
            raise ValueError(f"Unsupported emotion format: {emotion}")

        if target_length > 0:
            if len(emotion_array) == target_length:
                return emotion_array
            elif len(emotion_array) == 1:
                return np.concatenate([emotion_array] * target_length, axis=0)
            elif len(emotion_array) > target_length:
                return emotion_array[:target_length]
            else:
                raise ValueError(
                    f"Emotion length {len(emotion_array)} cannot match target {target_length}"
                )
        return emotion_array

    def __call__(self, audio_features: np.ndarray, start_index: int, emotion=None) -> np.ndarray:
        """Build the complete conditioning vector for a sequence of frames.

        Args:
            audio_features: HuBERT features, shape (num_frames, 1024).
            start_index: Global frame index for the start of this window.
                Negative values indicate padding frames (before audio starts).
            emotion: Optional per-call emotion override.

        Returns:
            Conditioning array, shape (num_frames, conditioning_dim).
            conditioning_dim = 1024 + 8 + 2 + 6 + 63 = 1103 (when all enabled).
        """
        num_frames = len(audio_features)
        conditioning_parts = [audio_features]

        # Emotion conditioning
        if self.use_emotion:
            if emotion is not None:
                emotion_features = self._parse_emotion(emotion, num_frames)
            elif self.emotion_window is not None and len(self.emotion_window) == num_frames:
                emotion_features = self.emotion_window
            else:
                # Index into per-frame emotion sequence with mirror looping
                frame_indices = [max(idx, 0) % self.num_emotion_frames
                                 for idx in range(start_index, start_index + num_frames)]
                emotion_features = self.emotion_sequence[frame_indices]
            conditioning_parts.append(emotion_features)

        # Eye openness conditioning
        if self.use_eye_open:
            if self.eye_open_window is not None and len(self.eye_open_window) == num_frames:
                eye_open_features = self.eye_open_window
            else:
                if self.eye_f0_mode:
                    frame_indices = [0] * num_frames
                else:
                    frame_indices = [mirror_index(max(idx, 0), self.num_eye_open_frames)
                                     for idx in range(start_index, start_index + num_frames)]
                eye_open_features = self.eye_open_sequence[frame_indices]
            conditioning_parts.append(eye_open_features)

        # Eye ball position conditioning
        if self.use_eye_ball:
            if self.eye_ball_window is not None and len(self.eye_ball_window) == num_frames:
                eye_ball_features = self.eye_ball_window
            else:
                if self.eye_f0_mode:
                    frame_indices = [0] * num_frames
                else:
                    frame_indices = [mirror_index(max(idx, 0), self.num_eye_ball_frames)
                                     for idx in range(start_index, start_index + num_frames)]
                eye_ball_features = self.eye_ball_sequence[frame_indices]
            conditioning_parts.append(eye_ball_features)

        # Shape coefficient conditioning (constant per avatar)
        if self.use_shape_coeff:
            if len(self.shape_coeff_window) == num_frames:
                shape_features = self.shape_coeff_window
            else:
                shape_features = np.stack([self.shape_coefficients] * num_frames, axis=0)
            conditioning_parts.append(shape_features)

        # Concatenate all conditioning signals along feature dimension
        if len(conditioning_parts) > 1:
            return np.concatenate(conditioning_parts, axis=-1)
        return audio_features
