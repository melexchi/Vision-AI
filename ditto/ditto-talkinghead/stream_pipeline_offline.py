"""Offline Streaming Pipeline: processes a full audio file against a source avatar.

This is the main pipeline orchestrator for offline (non-realtime) generation.
Given a source image/video and an audio file, it produces an animated video
by running the full Ditto pipeline:

    Audio → HuBERT features → LMDM diffusion → motion keypoints
    → stitch with source → warp 3D features → SPADE decode → composite

The pipeline uses a multi-threaded architecture with bounded queues:
    audio2motion_worker → gpu_worker (stitch+warp+decode) → putback_worker → writer_worker

Each worker runs in its own thread; GPU operations are merged into a single
thread to avoid GIL contention on CUDA calls.
"""

import threading
import queue
import numpy as np
import traceback
from tqdm import tqdm

from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.motion_stitch import MotionStitch
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack
from core.atomic_components.writer import VideoWriterByImageIO
from core.atomic_components.wav2feat import Wav2Feat
from core.atomic_components.cfg import parse_cfg, print_cfg


class StreamSDK:
    """Offline pipeline SDK: generates animated video from source + audio.

    Manages the full pipeline lifecycle: avatar registration, component setup,
    multi-threaded processing, and video output.
    """

    def __init__(self, cfg_pkl: str, data_root: str, **kwargs):
        # Parse config and instantiate all pipeline components
        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)

        self.default_kwargs = default_kwargs

        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()
        self.wav2feat = Wav2Feat(**wav2feat_cfg)

    def _merge_kwargs(self, default_kwargs: dict, run_kwargs: dict) -> dict:
        """Fill run_kwargs with defaults for any missing keys."""
        for key, value in default_kwargs.items():
            if key not in run_kwargs:
                run_kwargs[key] = value
        return run_kwargs

    def setup_Nd(self, num_driving_frames: int = -1, fade_in: int = -1,
                 fade_out: int = -1, ctrl_info=None, N_d: int = None):
        """Set the total number of driving frames and configure fade effects.

        Args:
            num_driving_frames: Total frames to generate.
            fade_in: Number of frames for fade-in effect (-1 = disabled).
            fade_out: Number of frames for fade-out effect (-1 = disabled).
            ctrl_info: Per-frame control dict (uses self.ctrl_info if None).
        """
        # Support legacy kwarg N_d= from callers
        if N_d is not None:
            num_driving_frames = N_d
        # Update motion stitch's total frame count (for eye blink scheduling)
        self.motion_stitch.set_Nd(num_driving_frames)

        if ctrl_info is None:
            ctrl_info = self.ctrl_info

        # Apply fade-in alpha ramp: 0.0 → 1.0 over fade_in frames
        if fade_in > 0:
            for frame_idx in range(fade_in):
                alpha = frame_idx / fade_in
                frame_ctrl = ctrl_info.get(frame_idx, {})
                frame_ctrl["fade_alpha"] = alpha
                ctrl_info[frame_idx] = frame_ctrl

        # Apply fade-out alpha ramp: 1.0 → 0.0 over fade_out frames
        if fade_out > 0:
            fade_start = num_driving_frames - fade_out
            fade_end = num_driving_frames - 1
            for frame_idx in range(fade_start, num_driving_frames):
                alpha = max((fade_end - frame_idx) / max(fade_end - fade_start, 1), 0)
                frame_ctrl = ctrl_info.get(frame_idx, {})
                frame_ctrl["fade_alpha"] = alpha
                ctrl_info[frame_idx] = frame_ctrl

        self.ctrl_info = ctrl_info

    def setup(self, source_path: str, output_path: str, **kwargs):
        """Full pipeline setup: register avatar, configure components, start workers.

        Args:
            source_path: Path to source image or video.
            output_path: Path for output video file.
            **kwargs: Pipeline configuration overrides (merged with defaults).
        """
        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        print("=" * 20, "setup kwargs", "=" * 20)
        print_cfg(**kwargs)
        print("=" * 50)

        self._extract_config(kwargs)

        # Only HuBERT supports online (streaming) mode
        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Register Avatar ========
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        num_source_frames = (
            self.template_num_frames if self.template_num_frames > 0
            else self.num_driving_frames
        )
        source_info = self.avatar_registrar(
            source_path,
            max_dim=self.max_size,
            n_frames=num_source_frames,
            **crop_kwargs,
        )

        # Apply temporal smoothing to video source motion parameters
        if len(source_info["x_s_info_lst"]) > 1 and self.source_smoothing_kernel > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(
                source_info["x_s_info_lst"], smo_k=self.source_smoothing_kernel,
            )

        self.source_info = source_info
        self.num_source_frames = len(source_info["x_s_info_lst"])

        # ======== Setup Components ========
        self._setup_components(source_info)

        # ======== Video Writer ========
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_progress = tqdm(desc="writer")

        # ======== Audio Feature Buffer ========
        self._init_audio_buffer()

        # ======== Start Worker Threads ========
        self._start_workers()

    def setup_from_cache(self, source_info: dict, output_path: str, **kwargs):
        """Setup pipeline using pre-computed source_info, skipping avatar registration.

        Used when the avatar has already been registered (e.g. from a previous
        session or cached on disk).

        Args:
            source_info: Pre-computed source_info dict from AvatarRegistrar.
            output_path: Path for output video file.
            **kwargs: Pipeline configuration overrides.
        """
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        self._extract_config(kwargs)

        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Use cached source_info (skip registration) ========
        import copy
        self.source_info = copy.deepcopy(source_info)
        self.num_source_frames = len(source_info["x_s_info_lst"])

        # ======== Setup Components ========
        self._setup_components(self.source_info)

        # ======== Cache source features on GPU for warp network ========
        if self.source_info["is_image_flag"]:
            self.warp_f3d.cache_source(self.source_info["f_s_lst"][0])

        # ======== Video Writer ========
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_progress = tqdm(desc="writer")

        # ======== Audio Feature Buffer ========
        self._init_audio_buffer()

        # ======== Start Worker Threads ========
        self._start_workers()

    def _extract_config(self, kwargs: dict):
        """Extract all config values from kwargs into instance attributes."""
        # -- Avatar registrar config --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_num_frames = kwargs.get("template_n_frames", -1)
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        self.source_smoothing_kernel = kwargs.get("smo_k_s", 13)

        # -- Condition handler config --
        self.emotion = kwargs.get("emo", 4)
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)
        self.condition_handler_info = kwargs.get("ch_info", None)

        # -- Audio2Motion (LMDM) config --
        self.overlap_frames = kwargs.get("overlap_v2", 10)
        self.keypoint_reset_interval = kwargs.get("fix_kp_cond", 0)
        self.keypoint_reset_dim_range = kwargs.get("fix_kp_cond_dim", None)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.value_clamp_range = kwargs.get("v_min_max_for_clip", None)
        self.driving_smoothing_kernel = kwargs.get("smo_k_d", 3)

        # -- Motion stitch config --
        self.num_driving_frames = kwargs.get("N_d", -1)
        self.driving_motion_keys = kwargs.get("use_d_keys", None)
        self.use_relative_driving = kwargs.get("relative_d", True)
        self.drive_eye_motion = kwargs.get("drive_eye", None)
        self.eye_delta_array = kwargs.get("delta_eye_arr", None)
        self.eye_open_delta_count = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.enable_stitching = kwargs.get("flag_stitching", True)

        # -- Per-frame control info --
        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())

    def _setup_components(self, source_info: dict):
        """Configure condition handler, audio2motion, and motion stitch."""
        # Condition handler: builds conditioning vectors from audio features
        self.condition_handler.setup(
            source_info, self.emotion,
            eye_f0_mode=self.eye_f0_mode,
            ch_info=self.condition_handler_info,
        )

        # Audio2Motion (LMDM): uses old kwarg names since they come from config pickle
        source_motion_params = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            source_motion_params,
            overlap_v2=self.overlap_frames,
            fix_kp_cond=self.keypoint_reset_interval,
            fix_kp_cond_dim=self.keypoint_reset_dim_range,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.value_clamp_range,
            smo_k_d=self.driving_smoothing_kernel,
        )

        # Motion stitch: merges source and driving motion parameters
        is_image = source_info["is_image_flag"]
        first_frame_motion = source_info["x_s_info_lst"][0]
        self.motion_stitch.setup(
            N_d=self.num_driving_frames,
            use_d_keys=self.driving_motion_keys,
            relative_d=self.use_relative_driving,
            drive_eye=self.drive_eye_motion,
            delta_eye_arr=self.eye_delta_array,
            delta_eye_open_n=self.eye_open_delta_count,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.enable_stitching,
            is_image_flag=is_image,
            x_s_info=first_frame_motion,
            d0=None,
            ch_info=self.condition_handler_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

    def _init_audio_buffer(self):
        """Initialize the audio feature buffer for LMDM input."""
        if self.online_mode:
            # Pre-fill buffer with silence features (overlap_frames worth)
            silence_samples = self.overlap_frames * 640  # 640 samples per frame at 16kHz
            self.audio_features = self.wav2feat.wav2feat(
                np.zeros((silence_samples,), dtype=np.float32), sr=16000,
            )
            assert len(self.audio_features) == self.overlap_frames
        else:
            self.audio_features = np.zeros(
                (0, self.wav2feat.feat_dim), dtype=np.float32,
            )
        # Offset for condition handler indexing (accounts for pre-filled silence)
        self.condition_index_start = -len(self.audio_features)

    def _start_workers(self, max_queue_size: int = 20):
        """Create and start pipeline worker threads."""
        self.worker_exception = None
        self.stop_event = threading.Event()

        self.audio2motion_queue = queue.Queue(maxsize=max_queue_size)
        self.motion_stitch_queue = queue.Queue(maxsize=max_queue_size)
        self.warp_f3d_queue = queue.Queue(maxsize=max_queue_size)
        self.decode_f3d_queue = queue.Queue(maxsize=max_queue_size)
        self.putback_queue = queue.Queue(maxsize=max_queue_size)
        self.writer_queue = queue.Queue(maxsize=max_queue_size)

        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker),
            threading.Thread(target=self.gpu_worker),
            threading.Thread(target=self.putback_worker),
            threading.Thread(target=self.writer_worker),
        ]

        for thread in self.thread_list:
            thread.start()

    def _get_ctrl_info(self, frame_idx: int) -> dict:
        """Get per-frame control kwargs for the given frame index."""
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(frame_idx, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[frame_idx]
            else:
                return {}
        except Exception:
            traceback.print_exc()
            return {}

    # ======== Worker Thread Entry Points ========
    # Each worker has a public method (exception handler) and private impl.

    def writer_worker(self):
        """Writer thread: writes composited frames to video file."""
        try:
            self._writer_worker_impl()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _writer_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break
            composited_frame = item
            self.writer(composited_frame, fmt="rgb")
            self.writer_progress.update()

    def putback_worker(self):
        """Putback thread: composites rendered face onto original background."""
        try:
            self._putback_worker_impl()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _putback_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.writer_queue.put(None)
                break
            source_frame_idx, rendered_face = item
            original_frame = self.source_info["img_rgb_lst"][source_frame_idx]
            crop_to_original = self.source_info["M_c2o_lst"][source_frame_idx]
            composited_frame = self.putback(original_frame, rendered_face, crop_to_original)
            self.writer_queue.put(composited_frame)

    def decode_f3d_worker(self):
        """Decode thread: runs SPADE decoder on warped 3D features."""
        try:
            self._decode_f3d_worker_impl()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _decode_f3d_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.putback_queue.put(None)
                break
            source_frame_idx, warped_features = item
            rendered_face = self.decode_f3d(warped_features)
            self.putback_queue.put([source_frame_idx, rendered_face])

    def warp_f3d_worker(self):
        """Warp thread: warps source 3D features to match driving keypoints."""
        try:
            self._warp_f3d_worker_impl()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _warp_f3d_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.decode_f3d_queue.put(None)
                break
            source_frame_idx, source_keypoints, driving_keypoints = item
            source_features = self.source_info["f_s_lst"][source_frame_idx]
            warped_features = self.warp_f3d(source_features, source_keypoints, driving_keypoints)
            self.decode_f3d_queue.put([source_frame_idx, warped_features])

    def gpu_worker(self):
        """Merged GPU worker: stitch + warp + decode in a single thread.

        Combining all GPU operations into one thread avoids GIL contention
        between CUDA kernel launches that would occur with separate threads.
        """
        try:
            self._gpu_worker_impl()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _gpu_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.putback_queue.put(None)
                break

            source_frame_idx, driving_motion, ctrl_kwargs = item

            # --- Stitch: merge source and driving motion ---
            source_motion = self.source_info["x_s_info_lst"][source_frame_idx]
            source_keypoints, driving_keypoints = self.motion_stitch(
                source_motion, driving_motion, **ctrl_kwargs,
            )

            # --- Warp: deform source features to match driving keypoints ---
            source_features = self.source_info["f_s_lst"][source_frame_idx]
            warped_features = self.warp_f3d(
                source_features, source_keypoints, driving_keypoints,
            )

            # --- Decode: render face image from warped features ---
            rendered_face = self.decode_f3d(warped_features)
            self.putback_queue.put([source_frame_idx, rendered_face])

    def motion_stitch_worker(self):
        """Stitch thread: merges source and driving motion parameters."""
        try:
            self._motion_stitch_worker_impl()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _motion_stitch_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.warp_f3d_queue.put(None)
                break
            source_frame_idx, driving_motion, ctrl_kwargs = item
            source_motion = self.source_info["x_s_info_lst"][source_frame_idx]
            source_kp, driving_kp = self.motion_stitch(
                source_motion, driving_motion, **ctrl_kwargs,
            )
            self.warp_f3d_queue.put([source_frame_idx, source_kp, driving_kp])

    def audio2motion_worker(self):
        """Audio-to-motion thread: converts audio features to motion keypoints."""
        try:
            self._audio2motion_offline()
        except Exception as exc:
            self.worker_exception = exc
            self.stop_event.set()

    def _audio2motion_offline(self):
        """Process full audio in one pass (offline mode).

        Waits for complete audio features from the queue, runs LMDM diffusion
        over the full sequence in sliding windows, then dispatches all motion
        frames to the stitch queue.
        """
        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break

            audio_features = item

            # Build conditioning vectors for all frames at once
            conditioning_sequence = self.condition_handler(audio_features, 0)
            window_size = self.audio2motion.seq_frames
            valid_frames_per_window = self.audio2motion.valid_clip_len
            total_frames = len(conditioning_sequence)

            # Run LMDM in sliding windows over the full conditioning sequence
            window_start = 0
            keypoint_sequence = None
            progress = tqdm(desc="dit")
            while window_start < total_frames:
                progress.update()

                # Extract window, padding the last window if needed
                window_cond = conditioning_sequence[
                    window_start:window_start + window_size
                ][None]  # Add batch dim: (1, window_size, cond_dim)
                if window_cond.shape[1] < window_size:
                    pad_frames = window_size - window_cond.shape[1]
                    padding = np.stack([window_cond[:, -1]] * pad_frames, axis=1)
                    window_cond = np.concatenate([window_cond, padding], axis=1)

                keypoint_sequence = self.audio2motion(window_cond, keypoint_sequence)
                window_start += valid_frames_per_window

            progress.close()

            # Trim to actual frame count and smooth
            keypoint_sequence = keypoint_sequence[:, :total_frames]
            keypoint_sequence = self.audio2motion._smo(
                keypoint_sequence, 0, keypoint_sequence.shape[1],
            )

            # Convert to per-frame motion parameter dicts
            driving_motion_list = self.audio2motion.cvt_fmt(keypoint_sequence)

            # Dispatch each frame to the stitch queue
            generated_frame_idx = 0
            for driving_motion in driving_motion_list:
                # Mirror-bounce through source frames for video templates
                source_frame_idx = _mirror_index(
                    generated_frame_idx, self.num_source_frames,
                )
                ctrl_kwargs = self._get_ctrl_info(generated_frame_idx)

                while not self.stop_event.is_set():
                    try:
                        self.motion_stitch_queue.put(
                            [source_frame_idx, driving_motion, ctrl_kwargs],
                            timeout=1,
                        )
                        break
                    except queue.Full:
                        continue
                generated_frame_idx += 1

            break  # Only process one audio input per session

        self.motion_stitch_queue.put(None)  # Signal end of stream

    def _audio2motion_worker(self):
        """Process streaming audio chunks incrementally (online mode).

        Buffers incoming audio features, runs LMDM in sliding windows as
        enough features accumulate, and dispatches motion frames in real-time.
        Periodically trims the feature buffer to prevent unbounded growth.
        """
        is_end = False
        window_size = self.audio2motion.seq_frames
        valid_frames_per_window = self.audio2motion.valid_clip_len
        feature_dim = self.wav2feat.feat_dim
        incoming_buffer = np.zeros((0, feature_dim), dtype=np.float32)

        keypoint_sequence = None
        # In online mode, first chunk initializes D0; valid_start is None until then
        valid_start_idx = None if self.online_mode else 0

        global_frame_idx = 0   # Absolute frame index (for condition handler)
        local_feature_idx = 0  # Index into self.audio_features buffer
        generated_frame_idx = 0

        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                is_end = True
            else:
                incoming_buffer = np.concatenate([incoming_buffer, item], axis=0)

            # Wait for at least valid_frames_per_window new features before processing
            if not is_end and incoming_buffer.shape[0] < valid_frames_per_window:
                continue
            else:
                self.audio_features = np.concatenate(
                    [self.audio_features, incoming_buffer], axis=0,
                )
                incoming_buffer = np.zeros((0, feature_dim), dtype=np.float32)

            # Process all complete windows in the current buffer
            while True:
                window_features = self.audio_features[
                    local_feature_idx:local_feature_idx + window_size
                ]
                actual_valid_len = valid_frames_per_window

                if len(window_features) == 0:
                    break
                elif len(window_features) < window_size:
                    if not is_end:
                        break  # Wait for more features
                    else:
                        # Final window: pad to full window size
                        actual_valid_len = len(window_features)
                        pad_count = window_size - len(window_features)
                        padding = np.stack([window_features[-1]] * pad_count, axis=0)
                        window_features = np.concatenate(
                            [window_features, padding], axis=0,
                        )

                # Build conditioning and run LMDM
                conditioning = self.condition_handler(
                    window_features,
                    global_frame_idx + self.condition_index_start,
                )
                keypoint_sequence = self.audio2motion(
                    conditioning, keypoint_sequence,
                )

                if valid_start_idx is None:
                    # Online mode, first chunk: initialize D0 reference pose
                    valid_start_idx = (
                        keypoint_sequence.shape[1] - self.audio2motion.fuse_length
                    )
                    first_frame_motion = self.audio2motion.cvt_fmt(
                        keypoint_sequence[0:1],
                    )[0]
                    self.motion_stitch.d0 = first_frame_motion

                    local_feature_idx += actual_valid_len
                    global_frame_idx += actual_valid_len
                    continue
                else:
                    # Extract valid frames from this window
                    valid_keypoints = keypoint_sequence[
                        :, valid_start_idx:valid_start_idx + actual_valid_len
                    ]
                    driving_motion_list = self.audio2motion.cvt_fmt(valid_keypoints)

                    # Dispatch to stitch queue
                    for driving_motion in driving_motion_list:
                        source_frame_idx = _mirror_index(
                            generated_frame_idx, self.num_source_frames,
                        )
                        ctrl_kwargs = self._get_ctrl_info(generated_frame_idx)

                        while not self.stop_event.is_set():
                            try:
                                self.motion_stitch_queue.put(
                                    [source_frame_idx, driving_motion, ctrl_kwargs],
                                    timeout=1,
                                )
                                break
                            except queue.Full:
                                continue
                        generated_frame_idx += 1

                    valid_start_idx += actual_valid_len
                    local_feature_idx += actual_valid_len
                    global_frame_idx += actual_valid_len

                # Trim keypoint buffer to prevent unbounded growth
                # Keep at most 2x window_size of history
                buffer_len = keypoint_sequence.shape[1]
                if buffer_len > window_size * 2:
                    trim_amount = buffer_len - window_size * 2
                    keypoint_sequence = keypoint_sequence[:, trim_amount:]
                    valid_start_idx -= trim_amount

                if local_feature_idx >= len(self.audio_features):
                    break

            # Trim audio feature buffer similarly
            feature_buffer_len = len(self.audio_features)
            if feature_buffer_len > window_size * 2:
                trim_amount = feature_buffer_len - window_size * 2
                self.audio_features = self.audio_features[trim_amount:]
                local_feature_idx -= trim_amount

            if is_end:
                break

        self.motion_stitch_queue.put(None)

    def close(self):
        """Flush pipeline, wait for all workers, and finalize output."""
        # Signal stop first so workers exit their loops, then send sentinel
        self.stop_event.set()
        self.audio2motion_queue.put(None)

        # Wait for all worker threads with timeout to prevent deadlock
        for thread in self.thread_list:
            thread.join(timeout=10)
            if thread.is_alive():
                print(f"WARNING: Worker thread {thread.name} did not exit within 10s")

        try:
            self.writer.close()
            self.writer_progress.close()
        except Exception:
            traceback.print_exc()

        # Re-raise any worker exception on the main thread
        if self.worker_exception is not None:
            raise self.worker_exception

    def run_chunk(self, audio_chunk: np.ndarray, chunksize: tuple = (3, 5, 2)):
        """Feed a streaming audio chunk into the pipeline.

        Only supported with HuBERT feature extractor.

        Args:
            audio_chunk: Raw audio samples (float32, 16kHz).
            chunksize: HuBERT streaming chunk config (left, center, right).
        """
        audio_features = self.wav2feat(audio_chunk, chunksize=chunksize)
        while not self.stop_event.is_set():
            try:
                self.audio2motion_queue.put(audio_features, timeout=1)
                break
            except queue.Full:
                continue
