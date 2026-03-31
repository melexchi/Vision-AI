"""Audio Waveform to Feature Extraction: converts raw audio into HuBERT features.

HuBERT (Hidden-Unit BERT) produces 1024-dim features at 50Hz (one feature per 20ms).
These features encode phonetic content that drives the LMDM motion model.

The streaming interface processes fixed-size audio chunks and extracts valid
features from the middle of each chunk (avoiding boundary artifacts from
HuBERT's transformer self-attention). The offline interface processes the
full audio in overlapping windows for maximum quality.
"""

import math
import numpy as np
import librosa

from ..aux_models.hubert_stream import HubertStreaming

# HuBERT output rate: 50 features/second (one per 20ms = 320 samples @16kHz)
HUBERT_FEATURE_DIM = 1024
HUBERT_HOP_SAMPLES = 320  # 16000 Hz / 50 features/sec
TARGET_SAMPLE_RATE = 16000
TARGET_FPS = 25  # Video frame rate — each frame uses 2 HuBERT features averaged


class Wav2Feat:
    """Unified audio-to-feature interface supporting HuBERT extraction.

    Wraps the streaming HuBERT model and provides both chunk-based (online)
    and full-audio (offline) feature extraction modes.
    """

    def __init__(self, w2f_cfg: dict, w2f_type: str = "hubert"):
        self.feature_type = w2f_type.lower()
        if self.feature_type == "hubert":
            self.extractor = Wav2FeatHubert(hubert_cfg=w2f_cfg)
            self.feat_dim = HUBERT_FEATURE_DIM
            self.support_streaming = True
        else:
            raise ValueError(f"Unsupported feature type: {w2f_type}")

    def __call__(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SAMPLE_RATE,
        norm_mean_std=None,
        chunksize: tuple[int, int, int] = (3, 5, 2),
    ) -> np.ndarray:
        """Extract features from a single audio chunk (streaming mode).

        Args:
            audio: Raw audio samples (float32, mono, 16kHz).
            chunksize: (prefix, valid, suffix) frame counts for HuBERT windowing.
                Each frame = 2 HuBERT features = 640 samples @16kHz.

        Returns:
            Feature array of shape (valid_frames, 1024).
        """
        if self.feature_type == "hubert":
            return self.extractor(audio, chunksize=chunksize)
        raise ValueError(f"Unsupported feature type: {self.feature_type}")

    def wav2feat(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SAMPLE_RATE,
        norm_mean_std=None,
        chunksize: tuple[int, int, int] = (3, 5, 2),
    ) -> np.ndarray:
        """Extract features from complete audio (offline mode).

        Processes the full audio in overlapping windows for consistent quality.
        Returns exactly ceil(audio_duration * 25) features (one per video frame).
        """
        if self.feature_type == "hubert":
            return self.extractor.extract_full_audio(audio, sample_rate=sr, chunksize=chunksize)
        raise ValueError(f"Unsupported feature type: {self.feature_type}")


class Wav2FeatHubert:
    """HuBERT-based audio feature extractor.

    HuBERT processes audio in overlapping windows. Each window is divided into
    three regions by the chunksize tuple (prefix, valid, suffix):
      - prefix: context frames before the valid region (needed for attention)
      - valid:  the frames whose features we actually keep
      - suffix: context frames after the valid region

    Each "frame" here = 2 HuBERT time steps = 640 audio samples @16kHz.
    So chunksize=(3,5,2) means 10 total frames = 6480 samples, yielding 5 valid features.
    """

    def __init__(self, hubert_cfg: dict):
        self.hubert = HubertStreaming(**hubert_cfg)

    def __call__(
        self, audio_chunk: np.ndarray, chunksize: tuple[int, int, int] = (3, 5, 2)
    ) -> np.ndarray:
        """Extract features from a single audio chunk (streaming mode).

        Args:
            audio_chunk: Audio samples, length = sum(chunksize) * 640 + 80.
                For default (3,5,2): 10 * 640 + 80 = 6480 samples.
            chunksize: (prefix_frames, valid_frames, suffix_frames).

        Returns:
            Feature array of shape (valid_frames, 1024). Each feature is the
            mean of 2 consecutive HuBERT outputs, aligning to 25fps video.
        """
        prefix_frames, valid_frames, suffix_frames = chunksize

        # HuBERT outputs 2 features per "frame", so valid region in HuBERT space:
        valid_start = -(valid_frames + suffix_frames) * 2  # e.g., -14
        valid_end = -suffix_frames * 2                      # e.g., -4

        hubert_encoding = self.hubert(audio_chunk)
        valid_encoding = hubert_encoding[valid_start:valid_end]

        # Average pairs of HuBERT features to get one feature per video frame (25fps)
        features = valid_encoding.reshape(valid_frames, 2, HUBERT_FEATURE_DIM).mean(axis=1)
        return features  # (valid_frames, 1024)

    def extract_full_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunksize: tuple[int, int, int] = (3, 5, 2),
    ) -> np.ndarray:
        """Extract features from complete audio in overlapping windows (offline mode).

        Pads the audio with silence at both ends so that every video frame gets
        proper context from HuBERT's attention mechanism.

        Returns exactly ceil(audio_duration_sec * 25) features.
        """
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        else:
            audio_16k = audio

        prefix_frames, valid_frames, suffix_frames = chunksize
        total_video_frames = math.ceil(len(audio_16k) / TARGET_SAMPLE_RATE * TARGET_FPS)

        # Total samples per HuBERT window
        samples_per_window = int(sum(chunksize) * 0.04 * TARGET_SAMPLE_RATE) + 80  # e.g., 6480

        # Pad audio: prefix silence + audio + suffix silence
        # The prefix silence ensures the first valid features start at audio onset
        suffix_context_samples = int((valid_frames + suffix_frames) * 0.04 * TARGET_SAMPLE_RATE)
        prefix_pad_samples = samples_per_window - suffix_context_samples
        padded_audio = np.concatenate([
            np.zeros(prefix_pad_samples, dtype=audio_16k.dtype),
            audio_16k,
            np.zeros(samples_per_window, dtype=audio_16k.dtype),
        ])

        # Process in overlapping windows, stepping by valid_frames
        all_features = []
        frame_idx = 0
        while frame_idx < total_video_frames:
            window_start = int(frame_idx * 0.04 * TARGET_SAMPLE_RATE)
            window_end = window_start + samples_per_window
            audio_window = padded_audio[window_start:window_end]

            window_features = self(audio_window, chunksize)
            all_features.append(window_features)
            frame_idx += valid_frames

        # Concatenate and trim to exact video frame count
        all_features = np.concatenate(all_features, axis=0)
        return all_features[:total_video_frames]
